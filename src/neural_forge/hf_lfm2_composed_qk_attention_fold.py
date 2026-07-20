"""Authenticated no-training composition of the LFM2 Q/K folding stages.

This module deliberately keeps the composition opt-in.  It does not alter the
historical attention candidates or the layer compiler.  The first stage:

* partitions the literal upstream FIT trajectories into local Q/K FIT and
  FIT-CHECK banks;
* compiles structural head assignments from Q/K-FIT only;
* uses CPCF only as an authenticated local factor extractor;
* synchronizes those FIT factors into one global dense Q/K seed;
* corrects and gates that seed against disjoint factor-derived target scores;
* rebuilds a structural producer with the accepted Q/K and FIT-only V planted
  byte-exactly.

The second stage cross-authenticates an externally compiled, fit-only residual
V/O artifact, converts it into the existing sequence-effective residual seed,
and returns an outer Q/K/V/O artifact.  Every numerical operation is an
analytical solve over cached evidence.  No model forward, backward, optimizer,
pruning, quantization, or target-template initialization dependence is
introduced here; template tensors contribute shape, dtype, and provenance only.

The current structural producer API uses one head dimension and one Q/K
RMSNorm contract for both source evidence and the target replay.  Consequently
this first production composition fails closed unless source and target head
dimensions and Q/K RMSNorm weights are identical.  CPCF itself already
supports head-dimension contraction; removing this composition-level
restriction requires a future structural-producer API with separate source
and target norm/head-dimension contracts.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from neural_forge import hf_lfm2_structural_residual_attention_fold as structural
from neural_forge.hf_lfm2_cached_qk_functional_refit import (
    Lfm2CachedQkEvidenceBank,
    adapt_lfm2_cached_attention_case_for_qk_refit,
    partition_lfm2_cached_qk_fit_cases,
)
from neural_forge.hf_lfm2_cached_qk_score_residual_fold import (
    Lfm2CompiledQkScoreResidualFold,
    Lfm2QkScoreResidualEvidenceBank,
    Lfm2QkScoreResidualFoldPolicy,
    build_lfm2_authenticated_dense_qk_seed,
    build_lfm2_qk_score_residual_case,
    compile_lfm2_cached_qk_score_residual_fold,
    seal_lfm2_qk_score_residual_fit_bank,
    seal_lfm2_qk_score_residual_fit_check_bank,
)
from neural_forge.hf_lfm2_cached_endpoint_qk_residual_fold import (
    Lfm2CompiledEndpointQkFitProposal,
    Lfm2DenseQkvoSourceInventory,
    Lfm2EndpointQkMetrics,
    Lfm2EndpointQkResidualCase,
    Lfm2EndpointQkResidualEvidenceBank,
    Lfm2EndpointQkResidualFoldPolicy,
    Lfm2EndpointQkvoSeed,
    build_lfm2_endpoint_qkvo_seed,
    compile_lfm2_cached_endpoint_qk_fit_proposal,
    evaluate_lfm2_endpoint_qkvo_bank,
    seal_lfm2_dense_qkvo_source_inventory,
    seal_lfm2_endpoint_qk_fit_bank,
    seal_lfm2_endpoint_qk_fit_check_bank,
)
from neural_forge.hf_lfm2_cpcf_attention_dimension_fold import (
    Lfm2CompiledCpcfAttentionDimensionFold,
    Lfm2CpcfAttentionCaseProvenance,
    Lfm2CpcfAttentionDimensionLimits,
    Lfm2CpcfAttentionDimensionPolicy,
    Lfm2CpcfAttentionDimensionProductionCase,
    Lfm2CpcfAttentionDimensionRuntime,
    Lfm2CpcfAttentionDimensionTopology,
    Lfm2CpcfBarycentricHeadGroup,
    compile_lfm2_cpcf_attention_dimension_fold,
)
from neural_forge.hf_lfm2_qk_global_gauge_sync import (
    Lfm2CompiledQkGlobalGaugeSync,
    Lfm2QkGlobalGaugeSyncPolicy,
    compile_lfm2_qk_global_gauge_sync,
    seal_lfm2_qk_global_gauge_sync_provenance,
)
from neural_forge.hf_lfm2_sequence_effective_attention_fold import (
    Lfm2CompiledFixedQkSequenceEffectiveAttentionFold,
    Lfm2FixedQkSequenceEffectiveVoAlsPolicy,
    Lfm2SequenceEffectiveResidualVoSeed,
    Lfm2SequenceEffectiveResidualVoSeedProvenance,
    compile_lfm2_fixed_qk_sequence_effective_attention_fold,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (
    Lfm2CachedAttentionFoldCase,
    Lfm2CompiledStructuralAttentionProducerFold,
    Lfm2CompiledStructuralAttentionValueOutputFold,
    compile_lfm2_structural_attention_producer_fold,
)


SCORE_ROUTED_RESIDUAL_SEEDED_SEQUENCE_EFFECTIVE_CANDIDATE_ID = (
    "score_routed_residual_seeded_sequence_effective"
)
ENDPOINT_RESIDUAL_ALTERNATING_QKVO_CANDIDATE_ID = "endpoint_residual_alternating_qkvo"
LFM2_COMPOSED_QK_PRODUCER_SCHEMA = "neural-forge-lfm2-composed-qk-attention-producer-v1"
LFM2_ENDPOINT_QK_START_SCHEMA = "neural-forge-lfm2-endpoint-qk-start-v1"
LFM2_COMPOSED_QK_STAGE_ONE_ATTEMPT_SCHEMA = (
    "neural-forge-lfm2-composed-qk-stage-one-attempt-v1"
)
LFM2_COMPOSED_QK_SEQUENCE_EFFECTIVE_SCHEMA = (
    "neural-forge-lfm2-composed-qk-sequence-effective-attention-v1"
)
LFM2_COMPOSED_ENDPOINT_RESIDUAL_POLICY_SCHEMA = (
    "neural-forge-lfm2-composed-endpoint-residual-policy-v1"
)
LFM2_COMPOSED_ENDPOINT_RESIDUAL_BLOCK_SCHEMA = (
    "neural-forge-lfm2-composed-endpoint-residual-block-v1"
)
LFM2_COMPOSED_ENDPOINT_RESIDUAL_SCHEMA = (
    "neural-forge-lfm2-composed-endpoint-residual-attention-v1"
)

_ZERO_OPERATION_FIELDS = (
    "model_forward_calls",
    "model_generate_calls",
    "backward_calls",
    "optimizer_steps",
    "training_steps",
    "pruned_parameters",
    "quantized_parameters",
    "selection_cases_consumed",
    "true_validation_cases_consumed",
    "trace_recaptures",
)


class Lfm2ComposedQkAttentionFoldError(ValueError):
    """The composed fold failed an evidence, quality, or authentication gate."""


class Lfm2ComposedQkAttentionAuthenticationError(Lfm2ComposedQkAttentionFoldError):
    """A nested artifact or composition seal is invalid."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional native dependency
        raise RuntimeError("PyTorch is required for composed LFM2 folding.") from exc
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


def _required_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2ComposedQkAttentionAuthenticationError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
    ):
        raise Lfm2ComposedQkAttentionFoldError(
            "Tensor hashing requires a dense non-meta tensor."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _structural_tensor_sha256(value: Any) -> str:
    """Match the structural/sequence artifact tensor digest exactly."""

    torch = _torch()
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _structural_state_sha256(values: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        digest.update(name.encode("utf-8"))
        digest.update(bytes.fromhex(_structural_tensor_sha256(values[name])))
    return digest.hexdigest()


def _case_bundle_sha256(cases: Sequence[Lfm2CachedAttentionFoldCase]) -> str:
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        if not isinstance(case, Lfm2CachedAttentionFoldCase):
            raise TypeError("upstream_fit_cases must contain cached attention cases.")
        tensors: dict[str, str | None] = {}
        for name in case.__dataclass_fields__:
            value = getattr(case, name)
            tensors[name] = None if value is None else _tensor_sha256(value)
        rows.append({"case_index": index, "tensors": tensors})
    if not rows:
        raise Lfm2ComposedQkAttentionFoldError(
            "At least one upstream FIT case is required."
        )
    return _canonical_sha256(rows)


def _upstream_trajectory_bundle(
    values: Sequence[str],
    *,
    expected_case_count: int,
) -> tuple[tuple[str, ...], str]:
    trajectories = tuple(
        _required_sha256(value, f"upstream_trajectory_sha256s[{index}]")
        for index, value in enumerate(values)
    )
    if (
        len(trajectories) != expected_case_count
        or len(set(trajectories)) != expected_case_count
    ):
        raise Lfm2ComposedQkAttentionAuthenticationError(
            "The ordered upstream trajectory inventory must contain exactly one "
            "distinct receipt for every upstream FIT case."
        )
    return trajectories, _canonical_sha256(list(trajectories))


def _ordered_partition_entries(
    fit_bank: Lfm2CachedQkEvidenceBank,
    fit_check_bank: Lfm2CachedQkEvidenceBank,
    *,
    expected_case_count: int,
) -> tuple[tuple[int, str, Lfm2CachedAttentionFoldCase], ...]:
    entries = tuple(
        (
            ordinal,
            trajectory,
            case,
        )
        for bank in (fit_bank, fit_check_bank)
        for ordinal, trajectory, case in zip(
            bank.source_case_ordinals,
            bank.trajectory_sha256s,
            bank.cases,
            strict=True,
        )
    )
    ordinals = tuple(value[0] for value in entries)
    trajectories = tuple(value[1] for value in entries)
    if (
        len(entries) != expected_case_count
        or sorted(ordinals) != list(range(expected_case_count))
        or len(set(trajectories)) != expected_case_count
    ):
        raise Lfm2ComposedQkAttentionAuthenticationError(
            "Q/K-FIT and FIT-CHECK do not form the exact complete upstream "
            "FIT trajectory partition."
        )
    return tuple(sorted(entries, key=lambda value: value[0]))


def _ordered_partition_cases(
    fit_bank: Lfm2CachedQkEvidenceBank,
    fit_check_bank: Lfm2CachedQkEvidenceBank,
    *,
    expected_case_count: int,
) -> tuple[Lfm2CachedAttentionFoldCase, ...]:
    return tuple(
        value[2]
        for value in _ordered_partition_entries(
            fit_bank,
            fit_check_bank,
            expected_case_count=expected_case_count,
        )
    )


def _ordered_partition_trajectory_sha256s(
    fit_bank: Lfm2CachedQkEvidenceBank,
    fit_check_bank: Lfm2CachedQkEvidenceBank,
    *,
    expected_case_count: int,
) -> tuple[str, ...]:
    return tuple(
        value[1]
        for value in _ordered_partition_entries(
            fit_bank,
            fit_check_bank,
            expected_case_count=expected_case_count,
        )
    )


def _factor_state_bundle_sha256(states: Sequence[Any]) -> str:
    return _canonical_sha256([value.as_digest_dict() for value in states])


def _structural_artifact_sha256(
    artifact: Lfm2CompiledStructuralAttentionProducerFold,
) -> str:
    return _canonical_sha256(artifact.as_dict())


def _value_output_artifact_sha256(
    artifact: Lfm2CompiledStructuralAttentionValueOutputFold,
) -> str:
    return _canonical_sha256(artifact.as_dict())


def _same_tensor(left: Any, right: Any) -> bool:
    torch = _torch()
    return (
        tuple(left.shape) == tuple(right.shape)
        and left.dtype == right.dtype
        and bool(torch.equal(left.detach().cpu(), right.detach().cpu()))
    )


def _positive_finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2ComposedQkAttentionFoldError(f"{name} must be finite and positive.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise Lfm2ComposedQkAttentionFoldError(
            f"{name} must be finite and positive."
        ) from exc
    if not math.isfinite(result) or result <= 0.0:
        raise Lfm2ComposedQkAttentionFoldError(f"{name} must be finite and positive.")
    return result


def _diagnostic_cpcf_policy() -> Lfm2CpcfAttentionDimensionPolicy:
    """CPCF is a factor extractor here; final Q/K gates are downstream."""

    return Lfm2CpcfAttentionDimensionPolicy(
        minimum_score_spectral_energy_fraction=0.0,
        maximum_fit_centered_score_relative_rmse=None,
        maximum_validation_centered_score_relative_rmse=None,
        maximum_validation_js_divergence=None,
        maximum_validation_route_probability_relative_rmse=None,
        minimum_validation_top1_route_agreement=None,
        anchor_exact_relative_rmse=0.01,
    )


def _strict_score_policy() -> Lfm2QkScoreResidualFoldPolicy:
    return Lfm2QkScoreResidualFoldPolicy(
        maximum_fit_centered_score_relative_rmse=0.10,
        maximum_fit_probability_relative_rmse=0.20,
        maximum_fit_js_divergence_mean=0.05,
        maximum_fit_js_divergence_worst=0.10,
        minimum_fit_top1_route_agreement=0.90,
        maximum_fit_check_centered_score_relative_rmse=0.20,
        maximum_fit_check_probability_relative_rmse=0.20,
        maximum_fit_check_js_divergence_mean=0.05,
        maximum_fit_check_js_divergence_worst=0.10,
        minimum_fit_check_top1_route_agreement=0.90,
    )


@dataclass(frozen=True, slots=True)
class Lfm2ComposedQkAttentionFoldPolicy:
    """Typed bounded policy for the complete analytical Q/K composition."""

    fit_check_count: int | None = None
    structural_ridge: float = 1e-8
    structural_cluster_ridge: float = 1e-8
    primitive_ridge: float = 1e-8
    cpcf_policy: Lfm2CpcfAttentionDimensionPolicy = field(
        default_factory=_diagnostic_cpcf_policy
    )
    cpcf_limits: Lfm2CpcfAttentionDimensionLimits | None = None
    global_policy: Lfm2QkGlobalGaugeSyncPolicy = field(
        default_factory=Lfm2QkGlobalGaugeSyncPolicy
    )
    score_policy: Lfm2QkScoreResidualFoldPolicy = field(
        default_factory=_strict_score_policy
    )
    authentication_sha256: str = ""

    def __post_init__(self) -> None:
        if self.fit_check_count is not None and (
            isinstance(self.fit_check_count, bool)
            or not isinstance(self.fit_check_count, int)
            or self.fit_check_count <= 0
        ):
            raise Lfm2ComposedQkAttentionFoldError(
                "fit_check_count must be a positive integer or None."
            )
        for name in (
            "structural_ridge",
            "structural_cluster_ridge",
            "primitive_ridge",
        ):
            object.__setattr__(
                self,
                name,
                _positive_finite(getattr(self, name), name),
            )
        if not isinstance(
            self.cpcf_policy,
            Lfm2CpcfAttentionDimensionPolicy,
        ):
            raise TypeError("cpcf_policy must be Lfm2CpcfAttentionDimensionPolicy.")
        if self.cpcf_limits is not None and not isinstance(
            self.cpcf_limits,
            Lfm2CpcfAttentionDimensionLimits,
        ):
            raise TypeError("cpcf_limits must be Lfm2CpcfAttentionDimensionLimits.")
        if not isinstance(self.global_policy, Lfm2QkGlobalGaugeSyncPolicy):
            raise TypeError("global_policy must be Lfm2QkGlobalGaugeSyncPolicy.")
        if not isinstance(self.score_policy, Lfm2QkScoreResidualFoldPolicy):
            raise TypeError("score_policy must be Lfm2QkScoreResidualFoldPolicy.")
        score = self.score_policy
        upper_bounds = {
            "maximum_fit_centered_score_relative_rmse": 0.10,
            "maximum_fit_probability_relative_rmse": 0.20,
            "maximum_fit_js_divergence_mean": 0.05,
            "maximum_fit_js_divergence_worst": 0.10,
            "maximum_fit_check_centered_score_relative_rmse": 0.20,
            "maximum_fit_check_probability_relative_rmse": 0.20,
            "maximum_fit_check_js_divergence_mean": 0.05,
            "maximum_fit_check_js_divergence_worst": 0.10,
        }
        if any(getattr(score, name) > limit for name, limit in upper_bounds.items()):
            raise Lfm2ComposedQkAttentionFoldError(
                "The composed score policy cannot relax its production quality "
                "ceilings."
            )
        if (
            score.minimum_fit_top1_route_agreement < 0.90
            or score.minimum_fit_check_top1_route_agreement < 0.90
        ):
            raise Lfm2ComposedQkAttentionFoldError(
                "The composed score policy cannot relax its top-1 route floor."
            )
        expected = _canonical_sha256(self.authentication_body())
        supplied = str(self.authentication_sha256).strip().lower()
        if supplied and supplied != expected:
            raise Lfm2ComposedQkAttentionAuthenticationError(
                "Composed Q/K policy authentication is invalid."
            )
        object.__setattr__(self, "authentication_sha256", expected)

    def authentication_body(self) -> dict[str, Any]:
        return {
            "schema": "neural-forge-lfm2-composed-qk-attention-policy-v1",
            "fit_check_count": self.fit_check_count,
            "structural_ridge": self.structural_ridge,
            "structural_cluster_ridge": self.structural_cluster_ridge,
            "primitive_ridge": self.primitive_ridge,
            "cpcf_policy": self.cpcf_policy.as_dict(),
            "cpcf_limits": (
                None if self.cpcf_limits is None else self.cpcf_limits.as_dict()
            ),
            "global_policy": self.global_policy.as_dict(),
            "score_policy": self.score_policy.as_dict(),
            "primitive_local_qk_published": False,
            "global_and_score_gates_authoritative": True,
            "training_allowed": False,
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
class Lfm2ComposedEndpointResidualAttentionFoldPolicy:
    """Bounded two-block Q/K endpoint correction and V/O refit policy."""

    maximum_outer_blocks: int = 2
    endpoint_policy: Lfm2EndpointQkResidualFoldPolicy = field(
        default_factory=Lfm2EndpointQkResidualFoldPolicy
    )
    sequence_policy: Lfm2FixedQkSequenceEffectiveVoAlsPolicy = field(
        default_factory=Lfm2FixedQkSequenceEffectiveVoAlsPolicy
    )
    allow_legacy_recomputed_parent_probabilities: bool = False
    authentication_sha256: str = ""

    def __post_init__(self) -> None:
        blocks = self.maximum_outer_blocks
        if (
            isinstance(blocks, bool)
            or not isinstance(blocks, int)
            or blocks
            not in {
                1,
                2,
            }
        ):
            raise Lfm2ComposedQkAttentionFoldError(
                "maximum_outer_blocks must be one or two."
            )
        if (
            not isinstance(
                self.endpoint_policy,
                Lfm2EndpointQkResidualFoldPolicy,
            )
            or not self.endpoint_policy.verify_authentication()
        ):
            raise TypeError(
                "endpoint_policy must be an authenticated endpoint Q/K policy."
            )
        if (
            not isinstance(
                self.sequence_policy,
                Lfm2FixedQkSequenceEffectiveVoAlsPolicy,
            )
            or not self.sequence_policy.verify_authentication()
        ):
            raise TypeError(
                "sequence_policy must be an authenticated sequence V/O policy."
            )
        if self.endpoint_policy.metric_regression_tolerance > 1e-4:
            raise Lfm2ComposedQkAttentionFoldError(
                "Endpoint FIT-CHECK tolerance exceeds the fifth-arm ceiling."
            )
        if not isinstance(
            self.allow_legacy_recomputed_parent_probabilities,
            bool,
        ):
            raise Lfm2ComposedQkAttentionFoldError(
                "allow_legacy_recomputed_parent_probabilities must be boolean."
            )
        expected = _canonical_sha256(self.authentication_body())
        supplied = str(self.authentication_sha256).strip().lower()
        if supplied and supplied != expected:
            raise Lfm2ComposedQkAttentionAuthenticationError(
                "Composed endpoint-residual policy authentication is invalid."
            )
        object.__setattr__(self, "authentication_sha256", expected)

    def authentication_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_COMPOSED_ENDPOINT_RESIDUAL_POLICY_SCHEMA,
            "maximum_outer_blocks": self.maximum_outer_blocks,
            "endpoint_policy": self.endpoint_policy.as_dict(),
            "sequence_policy": self.sequence_policy.as_dict(),
            "allow_legacy_recomputed_parent_probabilities": (
                self.allow_legacy_recomputed_parent_probabilities
            ),
            "outer_block_order": [
                "fixed_vo_endpoint_qk_fit",
                "fixed_qk_sequence_effective_vo_refit",
            ],
            "fit_check_can_trigger_refit": False,
            "training_allowed": False,
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


def _verify_structural_producer(
    artifact: Lfm2CompiledStructuralAttentionProducerFold,
) -> None:
    if not isinstance(artifact, Lfm2CompiledStructuralAttentionProducerFold):
        raise TypeError("Expected a structural attention producer artifact.")
    if (
        not artifact.full_source_coverage
        or _structural_state_sha256(artifact.state_dict())
        != artifact.compiled_state_sha256
        or artifact.target_logical_parameters >= artifact.source_logical_parameters
    ):
        raise Lfm2ComposedQkAttentionAuthenticationError(
            "Structural producer authentication or dense contraction failed."
        )


def _convert_head_groups(
    groups: Sequence[Sequence[int]],
    weights: Sequence[Sequence[float]],
) -> tuple[Lfm2CpcfBarycentricHeadGroup, ...]:
    if len(groups) != len(weights):
        raise Lfm2ComposedQkAttentionFoldError(
            "Structural groups and barycentric weights are misaligned."
        )
    return tuple(
        Lfm2CpcfBarycentricHeadGroup(
            source_indices=tuple(group),
            weights=tuple(float(value) for value in group_weights),
        )
        for group, group_weights in zip(groups, weights, strict=True)
    )


def _adapt_bank(
    bank: Lfm2CachedQkEvidenceBank,
    *,
    topology: Lfm2CpcfAttentionDimensionTopology,
    runtime: Lfm2CpcfAttentionDimensionRuntime,
) -> tuple[Lfm2CpcfAttentionDimensionProductionCase, ...]:
    return tuple(
        adapt_lfm2_cached_attention_case_for_qk_refit(
            case,
            topology=topology,
            runtime=runtime,
            provenance=Lfm2CpcfAttentionCaseProvenance(
                partition=bank.partition,
                trajectory_sha256=trajectory,
                source_split_sha256=bank.source_split_sha256,
                source_runtime_sha256=bank.source_runtime_sha256,
                case_ordinal=ordinal,
            ),
            case_name=f"{bank.partition}.cases[{index}]",
        )
        for index, (case, trajectory, ordinal) in enumerate(
            zip(
                bank.cases,
                bank.trajectory_sha256s,
                bank.source_case_ordinals,
                strict=True,
            )
        )
    )


def _score_cases(
    upstream_fold: Lfm2CompiledCpcfAttentionDimensionFold,
    production_cases: Sequence[Lfm2CpcfAttentionDimensionProductionCase],
) -> tuple[Any, ...]:
    state_members = {
        (state.partition, state.case_ordinal, state.evidence_sha256): state
        for state in upstream_fold.factor_sufficient_states
    }
    if len(state_members) != len(upstream_fold.factor_sufficient_states):
        raise Lfm2ComposedQkAttentionAuthenticationError(
            "The primitive published duplicate factor-state identities."
        )
    result = []
    for production in production_cases:
        key = (
            production.provenance.partition,
            production.provenance.case_ordinal,
            production.evidence_sha256,
        )
        if key not in state_members:
            raise Lfm2ComposedQkAttentionAuthenticationError(
                "A production case has no exact primitive factor-state member."
            )
        result.append(
            build_lfm2_qk_score_residual_case(
                production_case=production,
                upstream_fold=upstream_fold,
                factor_state=state_members[key],
            )
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class Lfm2CompiledComposedQkAttentionProducerFold:
    """Authenticated corrected dense structural producer and nested receipts."""

    policy: Lfm2ComposedQkAttentionFoldPolicy
    qk_fit_bank: Lfm2CachedQkEvidenceBank
    qk_fit_check_bank: Lfm2CachedQkEvidenceBank
    qk_fit_structural_producer: Lfm2CompiledStructuralAttentionProducerFold
    primitive_fold: Lfm2CompiledCpcfAttentionDimensionFold
    global_sync: Lfm2CompiledQkGlobalGaugeSync
    score_fit_bank: Lfm2QkScoreResidualEvidenceBank
    score_fit_check_bank: Lfm2QkScoreResidualEvidenceBank
    score_result: Lfm2CompiledQkScoreResidualFold
    corrected_producer: Lfm2CompiledStructuralAttentionProducerFold
    upstream_fit_case_count: int
    upstream_fit_case_bundle_sha256: str
    upstream_trajectory_sha256s: tuple[str, ...]
    upstream_trajectory_bundle_sha256: str
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.policy, Lfm2ComposedQkAttentionFoldPolicy)
            or not self.policy.verify_authentication()
        ):
            raise Lfm2ComposedQkAttentionAuthenticationError(
                "Composed Q/K policy authentication failed."
            )
        if (
            not isinstance(self.qk_fit_bank, Lfm2CachedQkEvidenceBank)
            or self.qk_fit_bank.partition != "fit"
            or not isinstance(self.qk_fit_check_bank, Lfm2CachedQkEvidenceBank)
            or self.qk_fit_check_bank.partition != "fit_check"
        ):
            raise Lfm2ComposedQkAttentionAuthenticationError(
                "The composed artifact lacks exact Q/K FIT/FIT-CHECK banks."
            )
        fit_sha = self.qk_fit_bank.verify()
        check_sha = self.qk_fit_check_bank.verify()
        case_count = self.upstream_fit_case_count
        if (
            isinstance(case_count, bool)
            or not isinstance(case_count, int)
            or case_count < 3
        ):
            raise Lfm2ComposedQkAttentionAuthenticationError(
                "The composed artifact requires at least three upstream FIT "
                "trajectories."
            )
        case_bundle = _required_sha256(
            self.upstream_fit_case_bundle_sha256,
            "upstream_fit_case_bundle_sha256",
        )
        upstream_trajectories, expected_trajectory_bundle = _upstream_trajectory_bundle(
            self.upstream_trajectory_sha256s,
            expected_case_count=case_count,
        )
        trajectory_bundle = _required_sha256(
            self.upstream_trajectory_bundle_sha256,
            "upstream_trajectory_bundle_sha256",
        )
        if trajectory_bundle != expected_trajectory_bundle:
            raise Lfm2ComposedQkAttentionAuthenticationError(
                "The upstream trajectory bundle seal does not authenticate its "
                "complete ordered receipt list."
            )
        if (
            self.qk_fit_bank.source_split_sha256
            != self.qk_fit_check_bank.source_split_sha256
            or self.qk_fit_bank.source_runtime_sha256
            != self.qk_fit_check_bank.source_runtime_sha256
            or set(self.qk_fit_bank.source_case_ordinals)
            & set(self.qk_fit_check_bank.source_case_ordinals)
            or set(self.qk_fit_bank.trajectory_sha256s)
            & set(self.qk_fit_check_bank.trajectory_sha256s)
        ):
            raise Lfm2ComposedQkAttentionAuthenticationError(
                "The local Q/K partitions overlap or name different evidence."
            )
        ordered_cases = _ordered_partition_cases(
            self.qk_fit_bank,
            self.qk_fit_check_bank,
            expected_case_count=case_count,
        )
        partition_trajectories = _ordered_partition_trajectory_sha256s(
            self.qk_fit_bank,
            self.qk_fit_check_bank,
            expected_case_count=case_count,
        )
        if partition_trajectories != upstream_trajectories:
            raise Lfm2ComposedQkAttentionAuthenticationError(
                "The Q/K FIT/FIT-CHECK banks do not reconstruct the exact "
                "ordered upstream trajectory inventory."
            )
        if _case_bundle_sha256(ordered_cases) != case_bundle:
            raise Lfm2ComposedQkAttentionAuthenticationError(
                "The full upstream FIT case receipt differs from the exact "
                "Q/K-FIT and FIT-CHECK partition union."
            )
        _verify_structural_producer(self.qk_fit_structural_producer)
        _verify_structural_producer(self.corrected_producer)
        primitive_sha = self.primitive_fold.verify_authentication()
        fit_states = tuple(
            state
            for state in self.primitive_fold.factor_sufficient_states
            if state.partition == "fit"
        )
        fit_state_sha = _factor_state_bundle_sha256(fit_states)
        primitive_factor_sha = _factor_state_bundle_sha256(
            self.primitive_fold.factor_sufficient_states
        )
        global_sha = self.global_sync.verify_authentication(
            expected_fit_receipt_sha256=primitive_sha,
            expected_source_runtime_receipt_sha256=(
                self.qk_fit_bank.source_runtime_sha256
            ),
            expected_factor_state_bundle_sha256=fit_state_sha,
        )
        score_fit_sha = self.score_fit_bank.verify()
        score_check_sha = self.score_fit_check_bank.verify()
        score_sha = self.score_result.verify()
        score_fit_receipts = tuple(
            (
                case.production_case.provenance.trajectory_sha256,
                case.production_case.provenance.case_ordinal,
            )
            for case in self.score_fit_bank.cases
        )
        score_check_receipts = tuple(
            (
                case.production_case.provenance.trajectory_sha256,
                case.production_case.provenance.case_ordinal,
            )
            for case in self.score_fit_check_bank.cases
        )
        qk_fit_receipts = tuple(
            zip(
                self.qk_fit_bank.trajectory_sha256s,
                self.qk_fit_bank.source_case_ordinals,
                strict=True,
            )
        )
        qk_check_receipts = tuple(
            zip(
                self.qk_fit_check_bank.trajectory_sha256s,
                self.qk_fit_check_bank.source_case_ordinals,
                strict=True,
            )
        )
        global_report = self.global_sync.as_dict()
        global_provenance = global_report.get("provenance")
        expected_prior_bundle = _canonical_sha256(
            {
                "source_query_weight_prior_sha256": _tensor_sha256(
                    self.qk_fit_structural_producer.query_weight
                ),
                "source_key_weight_prior_sha256": _tensor_sha256(
                    self.qk_fit_structural_producer.key_weight
                ),
            }
        )
        if (
            self.score_fit_bank.partition != "fit"
            or self.score_fit_check_bank.partition != "fit_check"
            or score_fit_receipts != qk_fit_receipts
            or score_check_receipts != qk_check_receipts
            or any(
                case.upstream_fold is not self.primitive_fold
                for case in (
                    *self.score_fit_bank.cases,
                    *self.score_fit_check_bank.cases,
                )
            )
            or self.score_fit_bank.source_artifact_sha256 != global_sha
            or self.score_fit_check_bank.source_artifact_sha256 != global_sha
            or self.score_result.seed.source_artifact_sha256 != global_sha
            or self.score_result.seed.upstream_fold is not self.primitive_fold
            or self.score_result.seed.global_sync is not self.global_sync
            or self.score_result.report.get("policy")
            != self.policy.score_policy.as_dict()
            or not isinstance(global_provenance, Mapping)
            or global_provenance.get("source_prior_bundle_sha256")
            != expected_prior_bundle
            or global_report.get("policy") != self.policy.global_policy.as_dict()
            or self.primitive_fold.report.get("policy")
            != self.policy.cpcf_policy.as_dict()
            or not self.score_result.accepted
            or not _same_tensor(
                self.score_result.query_weight,
                self.corrected_producer.query_weight,
            )
            or not _same_tensor(
                self.score_result.key_weight,
                self.corrected_producer.key_weight,
            )
            or not _same_tensor(
                self.qk_fit_structural_producer.value_weight,
                self.corrected_producer.value_weight,
            )
            or self.qk_fit_structural_producer.query_groups
            != self.corrected_producer.query_groups
            or self.qk_fit_structural_producer.query_barycentric_weights
            != self.corrected_producer.query_barycentric_weights
            or self.qk_fit_structural_producer.key_value_groups
            != self.corrected_producer.key_value_groups
            or self.qk_fit_structural_producer.key_value_barycentric_weights
            != self.corrected_producer.key_value_barycentric_weights
        ):
            raise Lfm2ComposedQkAttentionAuthenticationError(
                "The accepted score result was not planted exactly into the "
                "FIT-only structural producer."
            )
        body = dict(self.report)
        seal = body.pop("artifact_sha256", None)
        expected = {
            "schema": LFM2_COMPOSED_QK_PRODUCER_SCHEMA,
            "candidate_id": (
                SCORE_ROUTED_RESIDUAL_SEEDED_SEQUENCE_EFFECTIVE_CANDIDATE_ID
            ),
            "policy": self.policy.as_dict(),
            "policy_sha256": self.policy.authentication_sha256,
            "source_split_sha256": self.qk_fit_bank.source_split_sha256,
            "source_runtime_sha256": self.qk_fit_bank.source_runtime_sha256,
            "upstream_fit_case_count": case_count,
            "upstream_fit_case_bundle_sha256": case_bundle,
            "upstream_trajectory_sha256s": list(upstream_trajectories),
            "upstream_trajectory_bundle_sha256": trajectory_bundle,
            "qk_fit_evidence_sha256": fit_sha,
            "qk_fit_check_evidence_sha256": check_sha,
            "qk_fit_source_ordinals": list(self.qk_fit_bank.source_case_ordinals),
            "qk_fit_check_source_ordinals": list(
                self.qk_fit_check_bank.source_case_ordinals
            ),
            "qk_fit_structural_producer_sha256": _structural_artifact_sha256(
                self.qk_fit_structural_producer
            ),
            "primitive_report_sha256": primitive_sha,
            "primitive_factor_state_bundle_sha256": primitive_factor_sha,
            "primitive_fit_factor_state_bundle_sha256": fit_state_sha,
            "global_sync_report_sha256": global_sha,
            "score_fit_evidence_sha256": score_fit_sha,
            "score_fit_check_evidence_sha256": score_check_sha,
            "score_result_report_sha256": score_sha,
            "corrected_producer_sha256": _structural_artifact_sha256(
                self.corrected_producer
            ),
            "corrected_query_weight_sha256": _tensor_sha256(
                self.corrected_producer.query_weight
            ),
            "corrected_key_weight_sha256": _tensor_sha256(
                self.corrected_producer.key_weight
            ),
            "corrected_value_weight_sha256": _tensor_sha256(
                self.corrected_producer.value_weight
            ),
            "primitive_local_qk_published": False,
            "fit_check_used_for_structural_groups_or_priors": False,
            "fit_check_used_for_score_solver_updates": False,
            "fit_check_used_for_frozen_quality_gate": True,
            "ordinary_dense_target_qkv": True,
            "operation_audit": {name: 0 for name in _ZERO_OPERATION_FIELDS},
        }
        if seal != _canonical_sha256(body) or any(
            body.get(name) != value for name, value in expected.items()
        ):
            raise Lfm2ComposedQkAttentionAuthenticationError(
                "The composed Q/K producer receipt is invalid."
            )
        object.__setattr__(self, "upstream_fit_case_count", case_count)
        object.__setattr__(self, "upstream_fit_case_bundle_sha256", case_bundle)
        object.__setattr__(
            self,
            "upstream_trajectory_sha256s",
            upstream_trajectories,
        )
        object.__setattr__(
            self,
            "upstream_trajectory_bundle_sha256",
            trajectory_bundle,
        )
        object.__setattr__(
            self,
            "report",
            MappingProxyType({**body, "artifact_sha256": seal}),
        )

    @property
    def candidate_id(self) -> str:
        return SCORE_ROUTED_RESIDUAL_SEEDED_SEQUENCE_EFFECTIVE_CANDIDATE_ID

    @property
    def producer_artifact(self) -> Lfm2CompiledStructuralAttentionProducerFold:
        return self.corrected_producer

    @property
    def topology(self) -> Any:
        return self.corrected_producer.topology

    @property
    def artifact_sha256(self) -> str:
        return str(self.report["artifact_sha256"])

    def state_dict(self) -> Mapping[str, Any]:
        return self.corrected_producer.state_dict()

    def verify_authentication(self) -> str:
        rebuilt = Lfm2CompiledComposedQkAttentionProducerFold(
            policy=self.policy,
            qk_fit_bank=self.qk_fit_bank,
            qk_fit_check_bank=self.qk_fit_check_bank,
            qk_fit_structural_producer=self.qk_fit_structural_producer,
            primitive_fold=self.primitive_fold,
            global_sync=self.global_sync,
            score_fit_bank=self.score_fit_bank,
            score_fit_check_bank=self.score_fit_check_bank,
            score_result=self.score_result,
            corrected_producer=self.corrected_producer,
            upstream_fit_case_count=self.upstream_fit_case_count,
            upstream_fit_case_bundle_sha256=self.upstream_fit_case_bundle_sha256,
            upstream_trajectory_sha256s=self.upstream_trajectory_sha256s,
            upstream_trajectory_bundle_sha256=(self.upstream_trajectory_bundle_sha256),
            report=self.report,
        )
        return rebuilt.artifact_sha256

    def as_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dict(self.report)))


@dataclass(frozen=True, slots=True)
class Lfm2AuthenticatedEndpointQkStart:
    """Authenticated global-gauge dense QKV start for endpoint folding.

    This artifact deliberately does not claim that the score-residual fourth
    arm passed its admission gate.  It publishes the shared FIT-only
    structural/global-gauge prefix so an independently gated endpoint arm can
    still be attempted when that optional fourth arm is rejected.
    """

    policy: Lfm2ComposedQkAttentionFoldPolicy
    qk_fit_bank: Lfm2CachedQkEvidenceBank
    qk_fit_check_bank: Lfm2CachedQkEvidenceBank
    qk_fit_structural_producer: Lfm2CompiledStructuralAttentionProducerFold
    primitive_fold: Lfm2CompiledCpcfAttentionDimensionFold
    global_sync: Lfm2CompiledQkGlobalGaugeSync
    score_fit_bank: Lfm2QkScoreResidualEvidenceBank
    score_fit_check_bank: Lfm2QkScoreResidualEvidenceBank
    score_result: Lfm2CompiledQkScoreResidualFold
    starting_producer: Lfm2CompiledStructuralAttentionProducerFold
    upstream_fit_case_count: int
    upstream_fit_case_bundle_sha256: str
    upstream_trajectory_sha256s: tuple[str, ...]
    upstream_trajectory_bundle_sha256: str
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.policy, Lfm2ComposedQkAttentionFoldPolicy)
            or not self.policy.verify_authentication()
        ):
            raise Lfm2ComposedQkAttentionAuthenticationError(
                "Endpoint Q/K start policy authentication failed."
            )
        if (
            not isinstance(self.qk_fit_bank, Lfm2CachedQkEvidenceBank)
            or self.qk_fit_bank.partition != "fit"
            or not isinstance(self.qk_fit_check_bank, Lfm2CachedQkEvidenceBank)
            or self.qk_fit_check_bank.partition != "fit_check"
        ):
            raise Lfm2ComposedQkAttentionAuthenticationError(
                "Endpoint Q/K start lacks exact FIT/FIT-CHECK banks."
            )
        fit_sha = self.qk_fit_bank.verify()
        check_sha = self.qk_fit_check_bank.verify()
        case_count = self.upstream_fit_case_count
        if (
            isinstance(case_count, bool)
            or not isinstance(case_count, int)
            or case_count < 3
        ):
            raise Lfm2ComposedQkAttentionAuthenticationError(
                "Endpoint Q/K start requires at least three upstream FIT cases."
            )
        case_bundle = _required_sha256(
            self.upstream_fit_case_bundle_sha256,
            "upstream_fit_case_bundle_sha256",
        )
        trajectories, expected_trajectory_bundle = _upstream_trajectory_bundle(
            self.upstream_trajectory_sha256s,
            expected_case_count=case_count,
        )
        trajectory_bundle = _required_sha256(
            self.upstream_trajectory_bundle_sha256,
            "upstream_trajectory_bundle_sha256",
        )
        if trajectory_bundle != expected_trajectory_bundle:
            raise Lfm2ComposedQkAttentionAuthenticationError(
                "Endpoint Q/K start trajectory bundle authentication failed."
            )
        ordered_cases = _ordered_partition_cases(
            self.qk_fit_bank,
            self.qk_fit_check_bank,
            expected_case_count=case_count,
        )
        partition_trajectories = _ordered_partition_trajectory_sha256s(
            self.qk_fit_bank,
            self.qk_fit_check_bank,
            expected_case_count=case_count,
        )
        if (
            partition_trajectories != trajectories
            or _case_bundle_sha256(ordered_cases) != case_bundle
            or self.qk_fit_bank.source_split_sha256
            != self.qk_fit_check_bank.source_split_sha256
            or self.qk_fit_bank.source_runtime_sha256
            != self.qk_fit_check_bank.source_runtime_sha256
        ):
            raise Lfm2ComposedQkAttentionAuthenticationError(
                "Endpoint Q/K start partitions do not reconstruct the exact "
                "upstream evidence."
            )
        _verify_structural_producer(self.qk_fit_structural_producer)
        _verify_structural_producer(self.starting_producer)
        primitive_sha = self.primitive_fold.verify_authentication()
        fit_states = tuple(
            state
            for state in self.primitive_fold.factor_sufficient_states
            if state.partition == "fit"
        )
        fit_state_sha = _factor_state_bundle_sha256(fit_states)
        primitive_factor_sha = _factor_state_bundle_sha256(
            self.primitive_fold.factor_sufficient_states
        )
        global_sha = self.global_sync.verify_authentication(
            expected_fit_receipt_sha256=primitive_sha,
            expected_source_runtime_receipt_sha256=(
                self.qk_fit_bank.source_runtime_sha256
            ),
            expected_factor_state_bundle_sha256=fit_state_sha,
        )
        score_fit_sha = self.score_fit_bank.verify()
        score_check_sha = self.score_fit_check_bank.verify()
        score_sha = self.score_result.verify()
        seed = build_lfm2_authenticated_dense_qk_seed(
            upstream_fold=self.primitive_fold,
            global_sync=self.global_sync,
        )
        score_fit_receipts = tuple(
            (
                case.production_case.provenance.trajectory_sha256,
                case.production_case.provenance.case_ordinal,
            )
            for case in self.score_fit_bank.cases
        )
        score_check_receipts = tuple(
            (
                case.production_case.provenance.trajectory_sha256,
                case.production_case.provenance.case_ordinal,
            )
            for case in self.score_fit_check_bank.cases
        )
        qk_fit_receipts = tuple(
            zip(
                self.qk_fit_bank.trajectory_sha256s,
                self.qk_fit_bank.source_case_ordinals,
                strict=True,
            )
        )
        qk_check_receipts = tuple(
            zip(
                self.qk_fit_check_bank.trajectory_sha256s,
                self.qk_fit_check_bank.source_case_ordinals,
                strict=True,
            )
        )
        if (
            self.score_fit_bank.partition != "fit"
            or self.score_fit_check_bank.partition != "fit_check"
            or score_fit_receipts != qk_fit_receipts
            or score_check_receipts != qk_check_receipts
            or self.score_fit_bank.source_artifact_sha256 != global_sha
            or self.score_fit_check_bank.source_artifact_sha256 != global_sha
            or self.score_result.seed.upstream_fold is not self.primitive_fold
            or self.score_result.seed.global_sync is not self.global_sync
            or self.score_result.seed.verify() != seed.verify()
            or self.score_result.report.get("policy")
            != self.policy.score_policy.as_dict()
            or not _same_tensor(seed.query_weight, self.starting_producer.query_weight)
            or not _same_tensor(seed.key_weight, self.starting_producer.key_weight)
            or not _same_tensor(
                self.qk_fit_structural_producer.value_weight,
                self.starting_producer.value_weight,
            )
            or self.qk_fit_structural_producer.query_groups
            != self.starting_producer.query_groups
            or self.qk_fit_structural_producer.query_barycentric_weights
            != self.starting_producer.query_barycentric_weights
            or self.qk_fit_structural_producer.key_value_groups
            != self.starting_producer.key_value_groups
            or self.qk_fit_structural_producer.key_value_barycentric_weights
            != self.starting_producer.key_value_barycentric_weights
        ):
            raise Lfm2ComposedQkAttentionAuthenticationError(
                "Endpoint Q/K start is not the exact structural/global-gauge "
                "dense QKV seed."
            )
        body = dict(self.report)
        seal = body.pop("artifact_sha256", None)
        expected = {
            "schema": LFM2_ENDPOINT_QK_START_SCHEMA,
            "policy": self.policy.as_dict(),
            "policy_sha256": self.policy.authentication_sha256,
            "source_split_sha256": self.qk_fit_bank.source_split_sha256,
            "source_runtime_sha256": self.qk_fit_bank.source_runtime_sha256,
            "upstream_fit_case_count": case_count,
            "upstream_fit_case_bundle_sha256": case_bundle,
            "upstream_trajectory_sha256s": list(trajectories),
            "upstream_trajectory_bundle_sha256": trajectory_bundle,
            "qk_fit_evidence_sha256": fit_sha,
            "qk_fit_check_evidence_sha256": check_sha,
            "qk_fit_source_ordinals": list(self.qk_fit_bank.source_case_ordinals),
            "qk_fit_check_source_ordinals": list(
                self.qk_fit_check_bank.source_case_ordinals
            ),
            "qk_fit_structural_producer_sha256": _structural_artifact_sha256(
                self.qk_fit_structural_producer
            ),
            "primitive_report_sha256": primitive_sha,
            "primitive_factor_state_bundle_sha256": primitive_factor_sha,
            "primitive_fit_factor_state_bundle_sha256": fit_state_sha,
            "global_sync_report_sha256": global_sha,
            "score_fit_evidence_sha256": score_fit_sha,
            "score_fit_check_evidence_sha256": score_check_sha,
            "score_result_report_sha256": score_sha,
            "score_result_accepted": self.score_result.accepted,
            "score_residual_fourth_arm_eligible": self.score_result.accepted,
            "starting_producer_sha256": _structural_artifact_sha256(
                self.starting_producer
            ),
            "starting_query_weight_sha256": _tensor_sha256(
                self.starting_producer.query_weight
            ),
            "starting_key_weight_sha256": _tensor_sha256(
                self.starting_producer.key_weight
            ),
            "starting_value_weight_sha256": _tensor_sha256(
                self.starting_producer.value_weight
            ),
            "starting_state_source": (
                "fit_structural_groups_plus_global_gauge_qk_plus_fit_only_v"
            ),
            "fit_check_used_for_structural_groups_or_priors": False,
            "fit_check_used_for_score_solver_updates": False,
            "fit_check_used_only_for_score_candidate_gate": True,
            "ordinary_dense_target_qkv": True,
            "operation_audit": {name: 0 for name in _ZERO_OPERATION_FIELDS},
        }
        if seal != _canonical_sha256(body) or any(
            body.get(name) != value for name, value in expected.items()
        ):
            raise Lfm2ComposedQkAttentionAuthenticationError(
                "Endpoint Q/K start receipt is invalid."
            )
        object.__setattr__(self, "upstream_fit_case_count", case_count)
        object.__setattr__(self, "upstream_fit_case_bundle_sha256", case_bundle)
        object.__setattr__(self, "upstream_trajectory_sha256s", trajectories)
        object.__setattr__(
            self,
            "upstream_trajectory_bundle_sha256",
            trajectory_bundle,
        )
        object.__setattr__(
            self,
            "report",
            MappingProxyType({**body, "artifact_sha256": seal}),
        )

    @property
    def producer_artifact(self) -> Lfm2CompiledStructuralAttentionProducerFold:
        return self.starting_producer

    @property
    def topology(self) -> Any:
        return self.starting_producer.topology

    @property
    def artifact_sha256(self) -> str:
        return str(self.report["artifact_sha256"])

    def verify_authentication(self) -> str:
        rebuilt = Lfm2AuthenticatedEndpointQkStart(
            policy=self.policy,
            qk_fit_bank=self.qk_fit_bank,
            qk_fit_check_bank=self.qk_fit_check_bank,
            qk_fit_structural_producer=self.qk_fit_structural_producer,
            primitive_fold=self.primitive_fold,
            global_sync=self.global_sync,
            score_fit_bank=self.score_fit_bank,
            score_fit_check_bank=self.score_fit_check_bank,
            score_result=self.score_result,
            starting_producer=self.starting_producer,
            upstream_fit_case_count=self.upstream_fit_case_count,
            upstream_fit_case_bundle_sha256=self.upstream_fit_case_bundle_sha256,
            upstream_trajectory_sha256s=self.upstream_trajectory_sha256s,
            upstream_trajectory_bundle_sha256=(self.upstream_trajectory_bundle_sha256),
            report=self.report,
        )
        return rebuilt.artifact_sha256

    def as_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dict(self.report)))


@dataclass(frozen=True, slots=True)
class Lfm2ComposedQkStageOneAttempt:
    """Truthful stage-one outcome with an unconditional endpoint start."""

    endpoint_start: Lfm2AuthenticatedEndpointQkStart
    composed_candidate: Lfm2CompiledComposedQkAttentionProducerFold | None
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint_start, Lfm2AuthenticatedEndpointQkStart):
            raise TypeError("endpoint_start must be authenticated.")
        start_sha = self.endpoint_start.verify_authentication()
        score_result = self.endpoint_start.score_result
        score_sha = score_result.verify()
        candidate = self.composed_candidate
        if candidate is None:
            candidate_sha = None
        else:
            if not isinstance(candidate, Lfm2CompiledComposedQkAttentionProducerFold):
                raise TypeError("composed_candidate has the wrong type.")
            candidate_sha = candidate.verify_authentication()
        selected_start_sha = candidate_sha or start_sha
        if (
            (candidate is not None) != score_result.accepted
            or (
                candidate is not None
                and (
                    candidate.score_result is not score_result
                    or candidate.qk_fit_bank is not self.endpoint_start.qk_fit_bank
                    or candidate.qk_fit_check_bank
                    is not self.endpoint_start.qk_fit_check_bank
                    or candidate.primitive_fold is not self.endpoint_start.primitive_fold
                    or candidate.global_sync is not self.endpoint_start.global_sync
                )
            )
        ):
            raise Lfm2ComposedQkAttentionAuthenticationError(
                "Stage-one attempt relabeled a rejected score result or changed "
                "its authenticated shared prefix."
            )
        body = dict(self.report)
        seal = body.pop("attempt_sha256", None)
        expected = {
            "schema": LFM2_COMPOSED_QK_STAGE_ONE_ATTEMPT_SCHEMA,
            "endpoint_start_artifact_sha256": selected_start_sha,
            "global_gauge_fallback_start_artifact_sha256": start_sha,
            "endpoint_start_source": (
                "accepted_score_residual_corrected_dense_qkv"
                if candidate is not None
                else "structural_global_gauge_dense_qkv"
            ),
            "score_result_report_sha256": score_sha,
            "score_result_status": score_result.status,
            "score_result_accepted": score_result.accepted,
            "composed_candidate_eligible": candidate is not None,
            "composed_candidate_artifact_sha256": candidate_sha,
            "failed_composed_candidate_omitted": candidate is None,
            "endpoint_start_issued_independently": True,
            "operation_audit": {name: 0 for name in _ZERO_OPERATION_FIELDS},
        }
        if seal != _canonical_sha256(body) or any(
            body.get(name) != value for name, value in expected.items()
        ):
            raise Lfm2ComposedQkAttentionAuthenticationError(
                "Stage-one attempt receipt is invalid."
            )
        object.__setattr__(
            self,
            "report",
            MappingProxyType({**body, "attempt_sha256": seal}),
        )

    @property
    def attempt_sha256(self) -> str:
        return str(self.report["attempt_sha256"])

    @property
    def selected_endpoint_start(
        self,
    ) -> (
        Lfm2CompiledComposedQkAttentionProducerFold
        | Lfm2AuthenticatedEndpointQkStart
    ):
        return self.composed_candidate or self.endpoint_start

    def verify_authentication(self) -> str:
        rebuilt = Lfm2ComposedQkStageOneAttempt(
            endpoint_start=self.endpoint_start,
            composed_candidate=self.composed_candidate,
            report=self.report,
        )
        return rebuilt.attempt_sha256

    def as_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dict(self.report)))


def compile_lfm2_composed_qk_stage_one_attempt(
    *,
    source_query_weight: Any,
    source_key_weight: Any,
    source_value_weight: Any,
    source_output_weight: Any,
    target_query_weight_template: Any,
    target_key_weight_template: Any,
    target_value_weight_template: Any,
    target_output_weight_template: Any,
    upstream_fit_cases: Sequence[Lfm2CachedAttentionFoldCase],
    trajectory_sha256s: Sequence[str],
    source_split_sha256: str,
    source_query_norm_weight: Any,
    source_key_norm_weight: Any,
    target_query_norm_weight: Any,
    target_key_norm_weight: Any,
    source_head_dim: int,
    target_head_dim: int,
    norm_epsilon: float = 1e-5,
    residual_assignment_sha256: str | None = None,
    policy: Lfm2ComposedQkAttentionFoldPolicy | None = None,
) -> Lfm2ComposedQkStageOneAttempt:
    """Compile the shared prefix and truthfully attempt the optional fourth arm."""

    torch = _torch()
    policy_value = policy or Lfm2ComposedQkAttentionFoldPolicy()
    if (
        not isinstance(policy_value, Lfm2ComposedQkAttentionFoldPolicy)
        or not policy_value.verify_authentication()
    ):
        raise TypeError("policy must be an authenticated composed Q/K policy.")
    cases = tuple(upstream_fit_cases)
    trajectories, trajectory_bundle = _upstream_trajectory_bundle(
        trajectory_sha256s,
        expected_case_count=len(cases),
    )
    source_split = _required_sha256(source_split_sha256, "source_split_sha256")
    if len(cases) < 3 or len(trajectories) != len(cases):
        raise Lfm2ComposedQkAttentionFoldError(
            "Composition requires at least three aligned upstream FIT "
            "trajectories so Q/K-FIT contains at least two global-gauge cases."
        )
    if (
        isinstance(source_head_dim, bool)
        or not isinstance(source_head_dim, int)
        or source_head_dim <= 0
        or isinstance(target_head_dim, bool)
        or not isinstance(target_head_dim, int)
        or target_head_dim <= 0
        or source_head_dim % 2
        or target_head_dim % 2
    ):
        raise Lfm2ComposedQkAttentionFoldError(
            "Source and target head dimensions must be positive even integers."
        )
    source_q_norm = source_query_norm_weight.detach().cpu().contiguous()
    source_k_norm = source_key_norm_weight.detach().cpu().contiguous()
    target_q_norm = target_query_norm_weight.detach().cpu().contiguous()
    target_k_norm = target_key_norm_weight.detach().cpu().contiguous()
    if (
        source_head_dim != target_head_dim
        or not _same_tensor(source_q_norm, target_q_norm)
        or not _same_tensor(source_k_norm, target_k_norm)
    ):
        raise Lfm2ComposedQkAttentionFoldError(
            "The current structural-producer API requires preserved head "
            "dimension and identical source/target Q/K RMSNorm weights."
        )
    target_query_template = target_query_weight_template.detach().cpu().contiguous()
    target_key_template = target_key_weight_template.detach().cpu().contiguous()
    if (
        target_query_template.dtype != torch.bfloat16
        or target_key_template.dtype != torch.bfloat16
        or source_q_norm.dtype != torch.bfloat16
        or source_k_norm.dtype != torch.bfloat16
        or target_q_norm.dtype != torch.bfloat16
        or target_k_norm.dtype != torch.bfloat16
    ):
        raise Lfm2ComposedQkAttentionFoldError(
            "The production global/score composition currently publishes exact "
            "dense BF16 Q/K and therefore requires BF16 Q/K templates and norms."
        )
    runtime = Lfm2CpcfAttentionDimensionRuntime(
        source_query_norm_weight=source_q_norm,
        source_key_norm_weight=source_k_norm,
        target_query_norm_weight=target_q_norm,
        target_key_norm_weight=target_k_norm,
        target_query_weight_template=target_query_template,
        target_key_weight_template=target_key_template,
        norm_epsilon=norm_epsilon,
        solver_mode="per_case_local_gauge",
    )
    qk_fit_bank, qk_fit_check_bank = partition_lfm2_cached_qk_fit_cases(
        cases,
        trajectory_sha256s=trajectories,
        source_split_sha256=source_split,
        source_runtime_sha256=runtime.source_runtime_sha256,
        fit_check_count=policy_value.fit_check_count,
    )
    common_structural = {
        "source_query_weight": source_query_weight,
        "source_key_weight": source_key_weight,
        "source_value_weight": source_value_weight,
        "source_output_weight": source_output_weight,
        "target_query_weight_template": target_query_weight_template,
        "target_key_weight_template": target_key_weight_template,
        "target_value_weight_template": target_value_weight_template,
        "target_output_weight_template": target_output_weight_template,
        "cases": qk_fit_bank.cases,
        "head_dim": source_head_dim,
        "query_norm_weight": source_q_norm,
        "key_norm_weight": source_k_norm,
        "residual_assignment_sha256": residual_assignment_sha256,
        "norm_epsilon": norm_epsilon,
        "ridge": policy_value.structural_ridge,
        "cluster_ridge": policy_value.structural_cluster_ridge,
    }
    qk_fit_producer = compile_lfm2_structural_attention_producer_fold(
        **common_structural,
        max_prior_delta_ratio=None,
    )
    _verify_structural_producer(qk_fit_producer)
    topology = Lfm2CpcfAttentionDimensionTopology(
        source_query_heads=qk_fit_producer.topology.source_query_heads,
        source_key_value_heads=qk_fit_producer.topology.source_key_value_heads,
        target_query_heads=qk_fit_producer.topology.target_query_heads,
        target_key_value_heads=qk_fit_producer.topology.target_key_value_heads,
        source_head_dim=source_head_dim,
        target_head_dim=target_head_dim,
    )
    query_groups = _convert_head_groups(
        qk_fit_producer.query_groups,
        qk_fit_producer.query_barycentric_weights,
    )
    key_value_groups = _convert_head_groups(
        qk_fit_producer.key_value_groups,
        qk_fit_producer.key_value_barycentric_weights,
    )
    fit_production = _adapt_bank(
        qk_fit_bank,
        topology=topology,
        runtime=runtime,
    )
    fit_check_production = _adapt_bank(
        qk_fit_check_bank,
        topology=topology,
        runtime=runtime,
    )
    primitive = compile_lfm2_cpcf_attention_dimension_fold(
        topology=topology,
        fit_cases=fit_production,
        validation_cases=(),
        fit_check_cases=fit_check_production,
        query_groups=query_groups,
        key_value_groups=key_value_groups,
        policy=policy_value.cpcf_policy,
        limits=policy_value.cpcf_limits,
        ridge=policy_value.primitive_ridge,
        runtime=runtime,
    )
    primitive_sha = primitive.verify_authentication()
    fit_states = tuple(
        state
        for state in primitive.factor_sufficient_states
        if state.partition == "fit"
    )
    if len(fit_states) < 2:
        raise Lfm2ComposedQkAttentionFoldError(
            "Global Q/K synchronization requires at least two Q/K-FIT factors."
        )
    global_provenance = seal_lfm2_qk_global_gauge_sync_provenance(
        fit_receipt_sha256=primitive_sha,
        source_runtime_receipt_sha256=runtime.source_runtime_sha256,
        factor_states=fit_states,
        source_query_weight_prior=qk_fit_producer.query_weight,
        source_key_weight_prior=qk_fit_producer.key_weight,
        target_query_norm_weight=target_q_norm,
        target_key_norm_weight=target_k_norm,
        target_query_weight_template=target_query_template,
        target_key_weight_template=target_key_template,
        norm_epsilon=norm_epsilon,
    )
    global_sync = compile_lfm2_qk_global_gauge_sync(
        factor_states=fit_states,
        target_query_norm_weight=target_q_norm,
        target_key_norm_weight=target_k_norm,
        target_query_weight_template=target_query_template,
        target_key_weight_template=target_key_template,
        source_query_weight_prior=qk_fit_producer.query_weight,
        source_key_weight_prior=qk_fit_producer.key_weight,
        norm_epsilon=norm_epsilon,
        provenance=global_provenance,
        policy=policy_value.global_policy,
    )
    global_sha = global_sync.verify_authentication(
        expected_fit_receipt_sha256=primitive_sha,
        expected_source_runtime_receipt_sha256=runtime.source_runtime_sha256,
        expected_factor_state_bundle_sha256=_factor_state_bundle_sha256(fit_states),
    )
    seed = build_lfm2_authenticated_dense_qk_seed(
        upstream_fold=primitive,
        global_sync=global_sync,
    )
    score_fit_cases = _score_cases(primitive, fit_production)
    score_fit_check_cases = _score_cases(primitive, fit_check_production)
    score_fit_bank = seal_lfm2_qk_score_residual_fit_bank(
        score_fit_cases,
        source_split_sha256=source_split,
        source_runtime_sha256=runtime.source_runtime_sha256,
        source_artifact_sha256=global_sha,
    )
    score_fit_check_bank = seal_lfm2_qk_score_residual_fit_check_bank(
        score_fit_check_cases,
        source_split_sha256=source_split,
        source_runtime_sha256=runtime.source_runtime_sha256,
        source_artifact_sha256=global_sha,
    )
    score_result = compile_lfm2_cached_qk_score_residual_fold(
        seed=seed,
        fit_bank=score_fit_bank,
        fit_check_bank=score_fit_check_bank,
        policy=policy_value.score_policy,
    )
    global_seed_producer = compile_lfm2_structural_attention_producer_fold(
        **common_structural,
        query_weight_prior=seed.query_weight,
        key_weight_prior=seed.key_weight,
        value_weight_prior=qk_fit_producer.value_weight,
        max_prior_delta_ratio=0.0,
    )
    _verify_structural_producer(global_seed_producer)
    if (
        not _same_tensor(seed.query_weight, global_seed_producer.query_weight)
        or not _same_tensor(seed.key_weight, global_seed_producer.key_weight)
        or not _same_tensor(
            qk_fit_producer.value_weight,
            global_seed_producer.value_weight,
        )
        or qk_fit_producer.query_groups != global_seed_producer.query_groups
        or qk_fit_producer.query_barycentric_weights
        != global_seed_producer.query_barycentric_weights
        or qk_fit_producer.key_value_groups != global_seed_producer.key_value_groups
        or qk_fit_producer.key_value_barycentric_weights
        != global_seed_producer.key_value_barycentric_weights
    ):
        raise Lfm2ComposedQkAttentionAuthenticationError(
            "Exact global-gauge Q/K/V planting or FIT-only head assignment changed."
        )
    score_sha = score_result.verify()
    case_bundle = _case_bundle_sha256(cases)
    start_body = {
        "schema": LFM2_ENDPOINT_QK_START_SCHEMA,
        "policy": policy_value.as_dict(),
        "policy_sha256": policy_value.authentication_sha256,
        "source_split_sha256": source_split,
        "source_runtime_sha256": runtime.source_runtime_sha256,
        "upstream_fit_case_count": len(cases),
        "upstream_fit_case_bundle_sha256": case_bundle,
        "upstream_trajectory_sha256s": list(trajectories),
        "upstream_trajectory_bundle_sha256": trajectory_bundle,
        "qk_fit_evidence_sha256": qk_fit_bank.evidence_sha256,
        "qk_fit_check_evidence_sha256": qk_fit_check_bank.evidence_sha256,
        "qk_fit_source_ordinals": list(qk_fit_bank.source_case_ordinals),
        "qk_fit_check_source_ordinals": list(qk_fit_check_bank.source_case_ordinals),
        "qk_fit_structural_producer_sha256": _structural_artifact_sha256(
            qk_fit_producer
        ),
        "primitive_report_sha256": primitive_sha,
        "primitive_factor_state_bundle_sha256": _factor_state_bundle_sha256(
            primitive.factor_sufficient_states
        ),
        "primitive_fit_factor_state_bundle_sha256": _factor_state_bundle_sha256(
            fit_states
        ),
        "global_sync_report_sha256": global_sha,
        "score_fit_evidence_sha256": score_fit_bank.evidence_sha256,
        "score_fit_check_evidence_sha256": score_fit_check_bank.evidence_sha256,
        "score_result_report_sha256": score_sha,
        "score_result_accepted": score_result.accepted,
        "score_residual_fourth_arm_eligible": score_result.accepted,
        "starting_producer_sha256": _structural_artifact_sha256(
            global_seed_producer
        ),
        "starting_query_weight_sha256": _tensor_sha256(
            global_seed_producer.query_weight
        ),
        "starting_key_weight_sha256": _tensor_sha256(global_seed_producer.key_weight),
        "starting_value_weight_sha256": _tensor_sha256(
            global_seed_producer.value_weight
        ),
        "starting_state_source": (
            "fit_structural_groups_plus_global_gauge_qk_plus_fit_only_v"
        ),
        "fit_check_used_for_structural_groups_or_priors": False,
        "fit_check_used_for_score_solver_updates": False,
        "fit_check_used_only_for_score_candidate_gate": True,
        "ordinary_dense_target_qkv": True,
        "operation_audit": {name: 0 for name in _ZERO_OPERATION_FIELDS},
    }
    endpoint_start = Lfm2AuthenticatedEndpointQkStart(
        policy=policy_value,
        qk_fit_bank=qk_fit_bank,
        qk_fit_check_bank=qk_fit_check_bank,
        qk_fit_structural_producer=qk_fit_producer,
        primitive_fold=primitive,
        global_sync=global_sync,
        score_fit_bank=score_fit_bank,
        score_fit_check_bank=score_fit_check_bank,
        score_result=score_result,
        starting_producer=global_seed_producer,
        upstream_fit_case_count=len(cases),
        upstream_fit_case_bundle_sha256=case_bundle,
        upstream_trajectory_sha256s=trajectories,
        upstream_trajectory_bundle_sha256=trajectory_bundle,
        report={
            **start_body,
            "artifact_sha256": _canonical_sha256(start_body),
        },
    )
    corrected_producer = None
    composed_candidate = None
    if score_result.accepted:
        corrected_producer = compile_lfm2_structural_attention_producer_fold(
            **common_structural,
            query_weight_prior=score_result.query_weight,
            key_weight_prior=score_result.key_weight,
            value_weight_prior=qk_fit_producer.value_weight,
            max_prior_delta_ratio=0.0,
        )
        _verify_structural_producer(corrected_producer)
        if (
            not _same_tensor(
                score_result.query_weight,
                corrected_producer.query_weight,
            )
            or not _same_tensor(
                score_result.key_weight,
                corrected_producer.key_weight,
            )
            or not _same_tensor(
                qk_fit_producer.value_weight,
                corrected_producer.value_weight,
            )
            or qk_fit_producer.query_groups != corrected_producer.query_groups
            or qk_fit_producer.query_barycentric_weights
            != corrected_producer.query_barycentric_weights
            or qk_fit_producer.key_value_groups
            != corrected_producer.key_value_groups
            or qk_fit_producer.key_value_barycentric_weights
            != corrected_producer.key_value_barycentric_weights
        ):
            raise Lfm2ComposedQkAttentionAuthenticationError(
                "Exact accepted Q/K/V prior planting or FIT-only head "
                "assignment changed."
            )
    if corrected_producer is not None:
        body = {
        "schema": LFM2_COMPOSED_QK_PRODUCER_SCHEMA,
        "candidate_id": SCORE_ROUTED_RESIDUAL_SEEDED_SEQUENCE_EFFECTIVE_CANDIDATE_ID,
        "policy": policy_value.as_dict(),
        "policy_sha256": policy_value.authentication_sha256,
        "source_split_sha256": source_split,
        "source_runtime_sha256": runtime.source_runtime_sha256,
        "upstream_fit_case_count": len(cases),
        "upstream_fit_case_bundle_sha256": case_bundle,
        "upstream_trajectory_sha256s": list(trajectories),
        "upstream_trajectory_bundle_sha256": trajectory_bundle,
        "qk_fit_evidence_sha256": qk_fit_bank.evidence_sha256,
        "qk_fit_check_evidence_sha256": qk_fit_check_bank.evidence_sha256,
        "qk_fit_source_ordinals": list(qk_fit_bank.source_case_ordinals),
        "qk_fit_check_source_ordinals": list(qk_fit_check_bank.source_case_ordinals),
        "qk_fit_structural_producer_sha256": _structural_artifact_sha256(
            qk_fit_producer
        ),
        "primitive_report_sha256": primitive_sha,
        "primitive_factor_state_bundle_sha256": _factor_state_bundle_sha256(
            primitive.factor_sufficient_states
        ),
        "primitive_fit_factor_state_bundle_sha256": _factor_state_bundle_sha256(
            fit_states
        ),
        "global_sync_report_sha256": global_sha,
        "score_fit_evidence_sha256": score_fit_bank.evidence_sha256,
        "score_fit_check_evidence_sha256": score_fit_check_bank.evidence_sha256,
        "score_result_report_sha256": score_sha,
        "corrected_producer_sha256": _structural_artifact_sha256(corrected_producer),
        "corrected_query_weight_sha256": _tensor_sha256(
            corrected_producer.query_weight
        ),
        "corrected_key_weight_sha256": _tensor_sha256(corrected_producer.key_weight),
        "corrected_value_weight_sha256": _tensor_sha256(
            corrected_producer.value_weight
        ),
        "primitive_local_qk_published": False,
        "fit_check_used_for_structural_groups_or_priors": False,
        "fit_check_used_for_score_solver_updates": False,
        "fit_check_used_for_frozen_quality_gate": True,
        "ordinary_dense_target_qkv": True,
        "operation_audit": {name: 0 for name in _ZERO_OPERATION_FIELDS},
        }
        composed_candidate = Lfm2CompiledComposedQkAttentionProducerFold(
            policy=policy_value,
            qk_fit_bank=qk_fit_bank,
            qk_fit_check_bank=qk_fit_check_bank,
            qk_fit_structural_producer=qk_fit_producer,
            primitive_fold=primitive,
            global_sync=global_sync,
            score_fit_bank=score_fit_bank,
            score_fit_check_bank=score_fit_check_bank,
            score_result=score_result,
            corrected_producer=corrected_producer,
            upstream_fit_case_count=len(cases),
            upstream_fit_case_bundle_sha256=case_bundle,
            upstream_trajectory_sha256s=trajectories,
            upstream_trajectory_bundle_sha256=trajectory_bundle,
            report={**body, "artifact_sha256": _canonical_sha256(body)},
        )
    attempt_body = {
        "schema": LFM2_COMPOSED_QK_STAGE_ONE_ATTEMPT_SCHEMA,
        "endpoint_start_artifact_sha256": (
            composed_candidate.artifact_sha256
            if composed_candidate is not None
            else endpoint_start.artifact_sha256
        ),
        "global_gauge_fallback_start_artifact_sha256": (
            endpoint_start.artifact_sha256
        ),
        "endpoint_start_source": (
            "accepted_score_residual_corrected_dense_qkv"
            if composed_candidate is not None
            else "structural_global_gauge_dense_qkv"
        ),
        "score_result_report_sha256": score_sha,
        "score_result_status": score_result.status,
        "score_result_accepted": score_result.accepted,
        "composed_candidate_eligible": composed_candidate is not None,
        "composed_candidate_artifact_sha256": (
            None
            if composed_candidate is None
            else composed_candidate.artifact_sha256
        ),
        "failed_composed_candidate_omitted": composed_candidate is None,
        "endpoint_start_issued_independently": True,
        "operation_audit": {name: 0 for name in _ZERO_OPERATION_FIELDS},
    }
    return Lfm2ComposedQkStageOneAttempt(
        endpoint_start=endpoint_start,
        composed_candidate=composed_candidate,
        report={
            **attempt_body,
            "attempt_sha256": _canonical_sha256(attempt_body),
        },
    )


def compile_lfm2_composed_qk_attention_producer_fold(
    **kwargs: Any,
) -> Lfm2CompiledComposedQkAttentionProducerFold:
    """Compile the fourth arm and fail closed when its own gate rejects it."""

    attempt = compile_lfm2_composed_qk_stage_one_attempt(**kwargs)
    candidate = attempt.composed_candidate
    if candidate is None:
        score_result = attempt.endpoint_start.score_result
        score_report = dict(score_result.report)
        raise Lfm2ComposedQkAttentionFoldError(
            "The synchronized Q/K state failed the authoritative score-residual "
            f"gate: {score_result.status}; "
            f"FIT={score_result.fit_metrics.as_dict()!r}; "
            "FIT-CHECK="
            f"{None if score_result.fit_check_metrics is None else score_result.fit_check_metrics.as_dict()!r}; "
            "proposed FIT-CHECK="
            f"{score_report.get('proposed_fit_check_metrics')!r}."
        )
    return candidate


Lfm2EndpointQkStartingArtifact = (
    Lfm2CompiledComposedQkAttentionProducerFold | Lfm2AuthenticatedEndpointQkStart
)


def _verify_endpoint_qk_start(
    artifact: Lfm2EndpointQkStartingArtifact,
) -> str:
    if type(artifact) not in {
        Lfm2CompiledComposedQkAttentionProducerFold,
        Lfm2AuthenticatedEndpointQkStart,
    }:
        raise TypeError(
            "Endpoint Q/K start must be an accepted composed candidate or an "
            "authenticated global-gauge fallback."
        )
    return artifact.verify_authentication()


def _endpoint_qk_start_binding(
    artifact: Lfm2EndpointQkStartingArtifact,
    artifact_sha256: str,
) -> Mapping[str, Any]:
    artifact_sha = _required_sha256(
        artifact_sha256,
        "endpoint_qk_start_artifact_sha256",
    )
    if type(artifact) is Lfm2CompiledComposedQkAttentionProducerFold:
        return MappingProxyType({"composed_qk_artifact_sha256": artifact_sha})
    return MappingProxyType(
        {
            "composed_qk_artifact_sha256": None,
            "endpoint_qk_start_artifact_sha256": artifact_sha,
            "endpoint_qk_start_artifact_schema": LFM2_ENDPOINT_QK_START_SCHEMA,
            "endpoint_qk_start_source": "structural_global_gauge_dense_qkv",
            "composed_score_residual_candidate_eligible": False,
            "composed_score_residual_result_sha256": (
                artifact.score_result.verify()
            ),
        }
    )


def _stage_two_runtime_contract(
    composed: Lfm2EndpointQkStartingArtifact,
    *,
    query_norm_weight: Any,
    key_norm_weight: Any,
    norm_epsilon: float,
) -> tuple[Any, Any, float]:
    torch = _torch()
    runtime = composed.primitive_fold.runtime
    if not isinstance(query_norm_weight, torch.Tensor) or not isinstance(
        key_norm_weight,
        torch.Tensor,
    ):
        raise TypeError("Stage-two Q/K norm weights must be tensors.")
    query_norm = query_norm_weight.detach().to(device="cpu").contiguous()
    key_norm = key_norm_weight.detach().to(device="cpu").contiguous()
    epsilon = _positive_finite(norm_epsilon, "norm_epsilon")
    if (
        not _same_tensor(query_norm, runtime.target_query_norm_weight)
        or not _same_tensor(key_norm, runtime.target_key_norm_weight)
        or epsilon != runtime.norm_epsilon
    ):
        raise Lfm2ComposedQkAttentionAuthenticationError(
            "Stage-two Q/K norm weights and epsilon must exactly match the "
            "authenticated stage-one target runtime."
        )
    return query_norm, key_norm, epsilon


def _value_output_evidence_receipts(
    *,
    composed: Lfm2EndpointQkStartingArtifact,
    cases: Sequence[Lfm2CachedAttentionFoldCase],
    query_norm_weight: Any,
    key_norm_weight: Any,
    norm_epsilon: float,
    actual_child_contexts: Sequence[Any] | None,
) -> Mapping[str, str | None]:
    torch = _torch()
    topology = composed.topology
    query_norm = structural._float_tensor(  # noqa: SLF001
        query_norm_weight,
        "query_norm_weight",
        ndim=1,
    )
    key_norm = structural._float_tensor(  # noqa: SLF001
        key_norm_weight,
        "key_norm_weight",
        ndim=1,
    )
    normalized_cases = tuple(
        structural._normalize_attention_value_output_case(  # noqa: SLF001
            case,
            case_index=index,
            producer=composed.producer_artifact,
            query_norm_weight=query_norm_weight,
            key_norm_weight=key_norm_weight,
            normalized_query_norm_weight=query_norm,
            normalized_key_norm_weight=key_norm,
            norm_epsilon=norm_epsilon,
        )
        for index, case in enumerate(cases)
    )
    live_value = structural._float_tensor(  # noqa: SLF001
        composed.producer_artifact.value_weight,
        "composed.producer_artifact.value_weight",
        ndim=2,
    )
    reconstructed_rows = structural._attention_context_rows_from_value(  # noqa: SLF001
        normalized_cases,
        live_value,
        topology,
    )
    endpoint_target = torch.cat(
        [value["endpoint"] for value in normalized_cases],
        dim=0,
    )
    actual_sha: str | None = None
    if actual_child_contexts is not None:
        contexts = tuple(actual_child_contexts)
        if len(contexts) != len(cases):
            raise Lfm2ComposedQkAttentionAuthenticationError(
                "Actual child contexts must align one-to-one with the exact "
                "upstream FIT cases."
            )
        context_width = topology.target_query_heads * topology.head_dim
        actual_rows = tuple(
            structural._actual_attention_context_rows(  # noqa: SLF001
                context,
                case=case,
                context_width=context_width,
                name=f"actual_child_contexts[{index}]",
            )
            for index, (context, case) in enumerate(zip(contexts, cases, strict=True))
        )
        actual_sha = _structural_state_sha256(
            {f"context.{index:04d}": value for index, value in enumerate(actual_rows)}
        )
    return MappingProxyType(
        {
            "live_value_sha256": _structural_tensor_sha256(live_value),
            "actual_child_context_sha256": actual_sha,
            "reconstructed_live_context_sha256": _structural_state_sha256(
                {
                    f"context.{index:04d}": value
                    for index, value in enumerate(reconstructed_rows)
                }
            ),
            "endpoint_target_sha256": _structural_tensor_sha256(endpoint_target),
        }
    )


def _verify_value_output_artifact(
    artifact: Lfm2CompiledStructuralAttentionValueOutputFold,
    *,
    composed: Lfm2EndpointQkStartingArtifact,
    cases: Sequence[Lfm2CachedAttentionFoldCase],
    query_norm_weight: Any,
    key_norm_weight: Any,
    norm_epsilon: float,
    actual_child_contexts: Sequence[Any] | None,
) -> tuple[str, Mapping[str, str | None]]:
    if not isinstance(
        artifact,
        Lfm2CompiledStructuralAttentionValueOutputFold,
    ):
        raise TypeError("fit_value_output_artifact must be a structural V/O artifact.")
    state_sha = _structural_state_sha256(artifact.state_dict())
    receipts = _value_output_evidence_receipts(
        composed=composed,
        cases=cases,
        query_norm_weight=query_norm_weight,
        key_norm_weight=key_norm_weight,
        norm_epsilon=norm_epsilon,
        actual_child_contexts=actual_child_contexts,
    )
    if (
        artifact.topology != composed.topology
        or artifact.case_count != len(cases)
        or artifact.fixed_query_key_sha256
        != composed.producer_artifact.score_state_sha256
        or state_sha != artifact.compiled_state_sha256
        or not artifact.fit_nonincreasing
        or artifact.live_value_sha256 != receipts["live_value_sha256"]
        or artifact.reconstructed_live_context_sha256
        != receipts["reconstructed_live_context_sha256"]
        or artifact.endpoint_target_sha256 != receipts["endpoint_target_sha256"]
        or (
            receipts["actual_child_context_sha256"] is not None
            and artifact.actual_child_context_sha256
            != receipts["actual_child_context_sha256"]
        )
    ):
        raise Lfm2ComposedQkAttentionAuthenticationError(
            "The fit-only V/O artifact is not cross-bound to the corrected "
            "producer, exact cases/contexts, or dense state."
        )
    return _value_output_artifact_sha256(artifact), receipts


@dataclass(frozen=True, slots=True)
class Lfm2CompiledComposedQkSequenceEffectiveAttentionFold:
    """Outer authenticated fourth-arm artifact delegating to sequence V/O."""

    composed_qk: Lfm2CompiledComposedQkAttentionProducerFold
    fit_value_output_artifact: Lfm2CompiledStructuralAttentionValueOutputFold
    residual_vo_seed: Lfm2SequenceEffectiveResidualVoSeed
    inner_artifact: Lfm2CompiledFixedQkSequenceEffectiveAttentionFold
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        composed_sha = self.composed_qk.verify_authentication()
        runtime = self.composed_qk.primitive_fold.runtime
        cases = _ordered_partition_cases(
            self.composed_qk.qk_fit_bank,
            self.composed_qk.qk_fit_check_bank,
            expected_case_count=self.composed_qk.upstream_fit_case_count,
        )
        vo_sha, vo_receipts = _verify_value_output_artifact(
            self.fit_value_output_artifact,
            composed=self.composed_qk,
            cases=cases,
            query_norm_weight=runtime.target_query_norm_weight,
            key_norm_weight=runtime.target_key_norm_weight,
            norm_epsilon=runtime.norm_epsilon,
            actual_child_contexts=None,
        )
        if (
            not isinstance(
                self.residual_vo_seed,
                Lfm2SequenceEffectiveResidualVoSeed,
            )
            or not self.residual_vo_seed.verify_authentication()
            or self.residual_vo_seed.provenance.source_artifact_seal_sha256 != vo_sha
            or self.residual_vo_seed.provenance.source_compiled_state_sha256
            != self.fit_value_output_artifact.compiled_state_sha256
            or not _same_tensor(
                self.residual_vo_seed.value_weight,
                self.fit_value_output_artifact.value_weight,
            )
            or not _same_tensor(
                self.residual_vo_seed.output_weight,
                self.fit_value_output_artifact.output_weight,
            )
        ):
            raise Lfm2ComposedQkAttentionAuthenticationError(
                "Residual V/O seed does not cross-authenticate its source artifact."
            )
        if (
            not isinstance(
                self.inner_artifact,
                Lfm2CompiledFixedQkSequenceEffectiveAttentionFold,
            )
            or not self.inner_artifact.verify_authentication()
            or self.inner_artifact.producer_artifact
            is not self.composed_qk.producer_artifact
            or self.inner_artifact.residual_vo_seed is not self.residual_vo_seed
            or not _same_tensor(
                self.inner_artifact.producer_artifact.query_weight,
                self.composed_qk.score_result.query_weight,
            )
            or not _same_tensor(
                self.inner_artifact.producer_artifact.key_weight,
                self.composed_qk.score_result.key_weight,
            )
        ):
            raise Lfm2ComposedQkAttentionAuthenticationError(
                "The inner sequence artifact changed the composed Q/K or V/O seed."
            )
        inner_report = dict(self.inner_artifact.report)
        inner_sha = _required_sha256(
            inner_report.get("report_sha256"),
            "inner_report_sha256",
        )
        body = dict(self.report)
        seal = body.pop("artifact_sha256", None)
        expected = {
            "schema": LFM2_COMPOSED_QK_SEQUENCE_EFFECTIVE_SCHEMA,
            "candidate_id": (
                SCORE_ROUTED_RESIDUAL_SEEDED_SEQUENCE_EFFECTIVE_CANDIDATE_ID
            ),
            "composed_qk_artifact_sha256": composed_sha,
            "fit_value_output_artifact_sha256": vo_sha,
            "fit_value_output_compiled_state_sha256": (
                self.fit_value_output_artifact.compiled_state_sha256
            ),
            "fit_value_output_live_value_sha256": (vo_receipts["live_value_sha256"]),
            "fit_value_output_actual_child_context_sha256": (
                self.fit_value_output_artifact.actual_child_context_sha256
            ),
            "fit_value_output_reconstructed_live_context_sha256": (
                vo_receipts["reconstructed_live_context_sha256"]
            ),
            "fit_value_output_endpoint_target_sha256": (
                vo_receipts["endpoint_target_sha256"]
            ),
            "residual_vo_seed_sha256": (self.residual_vo_seed.authentication_sha256),
            "inner_sequence_report_sha256": inner_sha,
            "inner_compiled_state_sha256": self.inner_artifact.compiled_state_sha256,
            "fixed_query_key_sha256": (
                self.composed_qk.producer_artifact.score_state_sha256
            ),
            "upstream_fit_case_bundle_sha256": (
                self.composed_qk.upstream_fit_case_bundle_sha256
            ),
            "upstream_trajectory_sha256s": list(
                self.composed_qk.upstream_trajectory_sha256s
            ),
            "upstream_trajectory_bundle_sha256": (
                self.composed_qk.upstream_trajectory_bundle_sha256
            ),
            "query_norm_weight_sha256": _tensor_sha256(
                runtime.target_query_norm_weight
            ),
            "key_norm_weight_sha256": _tensor_sha256(runtime.target_key_norm_weight),
            "norm_epsilon": runtime.norm_epsilon,
            "dense_target_native_qkvo": True,
            "source_seed_fit_provenance_cross_authenticated": True,
            "operation_audit": {name: 0 for name in _ZERO_OPERATION_FIELDS},
        }
        if seal != _canonical_sha256(body) or any(
            body.get(name) != value for name, value in expected.items()
        ):
            raise Lfm2ComposedQkAttentionAuthenticationError(
                "The outer composed sequence-effective receipt is invalid."
            )
        object.__setattr__(
            self,
            "report",
            MappingProxyType({**body, "artifact_sha256": seal}),
        )

    @property
    def candidate_id(self) -> str:
        return SCORE_ROUTED_RESIDUAL_SEEDED_SEQUENCE_EFFECTIVE_CANDIDATE_ID

    @property
    def producer_artifact(self) -> Lfm2CompiledStructuralAttentionProducerFold:
        return self.inner_artifact.producer_artifact

    @property
    def topology(self) -> Any:
        return self.inner_artifact.topology

    @property
    def cluster_plan(self) -> Any:
        return self.inner_artifact.cluster_plan

    @property
    def artifact_sha256(self) -> str:
        return str(self.report["artifact_sha256"])

    def state_dict(self) -> Mapping[str, Any]:
        return self.inner_artifact.state_dict()

    def verify_authentication(self) -> str:
        rebuilt = Lfm2CompiledComposedQkSequenceEffectiveAttentionFold(
            composed_qk=self.composed_qk,
            fit_value_output_artifact=self.fit_value_output_artifact,
            residual_vo_seed=self.residual_vo_seed,
            inner_artifact=self.inner_artifact,
            report=self.report,
        )
        return rebuilt.artifact_sha256

    def as_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dict(self.report)))


def compile_lfm2_composed_qk_sequence_effective_attention_fold(
    *,
    composed_qk: Lfm2CompiledComposedQkAttentionProducerFold,
    fit_value_output_artifact: Lfm2CompiledStructuralAttentionValueOutputFold,
    upstream_fit_cases: Sequence[Lfm2CachedAttentionFoldCase],
    actual_child_contexts: Sequence[Any],
    target_value_weight_template: Any,
    target_output_weight_template: Any,
    query_norm_weight: Any,
    key_norm_weight: Any,
    norm_epsilon: float = 1e-5,
    sequence_policy: Lfm2FixedQkSequenceEffectiveVoAlsPolicy | None = None,
) -> Lfm2CompiledComposedQkSequenceEffectiveAttentionFold:
    """Cross-authenticate fit-only V/O and compile the fourth attention arm."""

    if not isinstance(composed_qk, Lfm2CompiledComposedQkAttentionProducerFold):
        raise TypeError("composed_qk must be a composed Q/K producer artifact.")
    composed_sha = composed_qk.verify_authentication()
    query_norm, key_norm, epsilon = _stage_two_runtime_contract(
        composed_qk,
        query_norm_weight=query_norm_weight,
        key_norm_weight=key_norm_weight,
        norm_epsilon=norm_epsilon,
    )
    cases = tuple(upstream_fit_cases)
    case_bundle = _case_bundle_sha256(cases)
    if (
        len(cases) != composed_qk.upstream_fit_case_count
        or case_bundle != composed_qk.upstream_fit_case_bundle_sha256
    ):
        raise Lfm2ComposedQkAttentionAuthenticationError(
            "Stage two must consume the exact full upstream FIT case bundle "
            "authenticated by stage one."
        )
    vo_sha, vo_receipts = _verify_value_output_artifact(
        fit_value_output_artifact,
        composed=composed_qk,
        cases=cases,
        query_norm_weight=query_norm,
        key_norm_weight=key_norm,
        norm_epsilon=epsilon,
        actual_child_contexts=actual_child_contexts,
    )
    provenance = Lfm2SequenceEffectiveResidualVoSeedProvenance(
        source_artifact_schema=(
            "neural-forge-lfm2-structural-attention-value-output-fold-v1"
        ),
        source_artifact_seal_sha256=vo_sha,
        source_compiled_state_sha256=(fit_value_output_artifact.compiled_state_sha256),
        source_value_weight_sha256=_structural_tensor_sha256(
            fit_value_output_artifact.value_weight
        ),
        source_output_weight_sha256=_structural_tensor_sha256(
            fit_value_output_artifact.output_weight
        ),
        evidence_split="fit",
    )
    residual_seed = Lfm2SequenceEffectiveResidualVoSeed(
        value_weight=fit_value_output_artifact.value_weight,
        output_weight=fit_value_output_artifact.output_weight,
        provenance=provenance,
    )
    inner = compile_lfm2_fixed_qk_sequence_effective_attention_fold(
        producer=composed_qk.producer_artifact,
        target_value_weight_template=target_value_weight_template,
        target_output_weight_template=target_output_weight_template,
        fit_cases=cases,
        query_norm_weight=query_norm,
        key_norm_weight=key_norm,
        norm_epsilon=epsilon,
        policy=sequence_policy,
        residual_vo_seed=residual_seed,
    )
    if (
        not inner.verify_authentication()
        or inner.producer_artifact is not composed_qk.producer_artifact
        or not _same_tensor(
            inner.producer_artifact.query_weight,
            composed_qk.score_result.query_weight,
        )
        or not _same_tensor(
            inner.producer_artifact.key_weight,
            composed_qk.score_result.key_weight,
        )
    ):
        raise Lfm2ComposedQkAttentionAuthenticationError(
            "Sequence-effective composition did not preserve exact corrected Q/K."
        )
    inner_sha = _required_sha256(
        dict(inner.report).get("report_sha256"),
        "inner_report_sha256",
    )
    body = {
        "schema": LFM2_COMPOSED_QK_SEQUENCE_EFFECTIVE_SCHEMA,
        "candidate_id": SCORE_ROUTED_RESIDUAL_SEEDED_SEQUENCE_EFFECTIVE_CANDIDATE_ID,
        "composed_qk_artifact_sha256": composed_sha,
        "fit_value_output_artifact_sha256": vo_sha,
        "fit_value_output_compiled_state_sha256": (
            fit_value_output_artifact.compiled_state_sha256
        ),
        "fit_value_output_live_value_sha256": vo_receipts["live_value_sha256"],
        "fit_value_output_actual_child_context_sha256": (
            vo_receipts["actual_child_context_sha256"]
        ),
        "fit_value_output_reconstructed_live_context_sha256": (
            vo_receipts["reconstructed_live_context_sha256"]
        ),
        "fit_value_output_endpoint_target_sha256": (
            vo_receipts["endpoint_target_sha256"]
        ),
        "residual_vo_seed_sha256": residual_seed.authentication_sha256,
        "inner_sequence_report_sha256": inner_sha,
        "inner_compiled_state_sha256": inner.compiled_state_sha256,
        "fixed_query_key_sha256": composed_qk.producer_artifact.score_state_sha256,
        "upstream_fit_case_bundle_sha256": case_bundle,
        "upstream_trajectory_sha256s": list(composed_qk.upstream_trajectory_sha256s),
        "upstream_trajectory_bundle_sha256": (
            composed_qk.upstream_trajectory_bundle_sha256
        ),
        "query_norm_weight_sha256": _tensor_sha256(query_norm),
        "key_norm_weight_sha256": _tensor_sha256(key_norm),
        "norm_epsilon": epsilon,
        "dense_target_native_qkvo": True,
        "source_seed_fit_provenance_cross_authenticated": True,
        "operation_audit": {name: 0 for name in _ZERO_OPERATION_FIELDS},
    }
    return Lfm2CompiledComposedQkSequenceEffectiveAttentionFold(
        composed_qk=composed_qk,
        fit_value_output_artifact=fit_value_output_artifact,
        residual_vo_seed=residual_seed,
        inner_artifact=inner,
        report={**body, "artifact_sha256": _canonical_sha256(body)},
    )


def _cached_parent_attention_probabilities(
    case: Lfm2CachedAttentionFoldCase,
) -> tuple[Any, Any, str, float, float, float, float]:
    torch = _torch()
    query = case.parent_query.detach().to(device="cpu", dtype=torch.float32)
    key = case.parent_key.detach().to(device="cpu", dtype=torch.float32)
    if (
        query.ndim != 4
        or key.ndim != 4
        or int(query.shape[0]) != int(key.shape[0])
        or int(query.shape[-1]) != int(key.shape[-1])
        or int(query.shape[1]) % int(key.shape[1])
    ):
        raise Lfm2ComposedQkAttentionFoldError(
            "Cached parent Q/K do not define one exact GQA probability route."
        )
    batch, heads, query_length, head_dim = map(int, query.shape)
    key_length = int(key.shape[2])
    if case.key_mask is None:
        offset = key_length - query_length
        if offset < 0:
            raise Lfm2ComposedQkAttentionFoldError(
                "Cached attention evidence has fewer keys than queries."
            )
        query_index = torch.arange(query_length).reshape(1, 1, -1, 1)
        key_index = torch.arange(key_length).reshape(1, 1, 1, -1)
        mask = (key_index <= offset + query_index).expand(
            batch,
            heads,
            query_length,
            key_length,
        )
    else:
        raw_mask = case.key_mask.detach().to(device="cpu")
        if raw_mask.dtype != torch.bool or raw_mask.ndim not in {3, 4}:
            raise Lfm2ComposedQkAttentionFoldError(
                "Cached attention key mask must be rank-3/4 boolean evidence."
            )
        if raw_mask.ndim == 3:
            raw_mask = raw_mask.unsqueeze(1)
        if (
            tuple(raw_mask.shape[0:1]) != (batch,)
            or tuple(raw_mask.shape[2:]) != (query_length, key_length)
            or int(raw_mask.shape[1]) not in {1, heads}
        ):
            raise Lfm2ComposedQkAttentionFoldError(
                "Cached attention key mask differs from parent Q/K geometry."
            )
        mask = (
            raw_mask.expand(-1, heads, -1, -1)
            if int(raw_mask.shape[1]) == 1
            else raw_mask
        ).contiguous()
    cached_probabilities = getattr(case, "parent_probabilities", None)
    if cached_probabilities is None:
        repeated_key = key.repeat_interleave(heads // int(key.shape[1]), dim=1)
        scores = torch.matmul(query, repeated_key.transpose(-2, -1)) / math.sqrt(
            head_dim
        )
        probabilities = torch.softmax(
            scores.masked_fill(~mask, -torch.inf),
            dim=-1,
        )
        probabilities = torch.where(
            mask,
            probabilities,
            torch.zeros_like(probabilities),
        )
        provenance = "legacy_recomputed_parent_qk"
    else:
        if (
            not isinstance(cached_probabilities, torch.Tensor)
            or cached_probabilities.layout != torch.strided
            or cached_probabilities.device.type == "meta"
            or cached_probabilities.ndim != 4
            or not cached_probabilities.dtype.is_floating_point
        ):
            raise Lfm2ComposedQkAttentionFoldError(
                "Cached parent attention probabilities must be one dense "
                "floating rank-4 tensor."
            )
        probabilities = cached_probabilities.detach().to(
            device="cpu",
            dtype=torch.float32,
        )
        if tuple(probabilities.shape) != (
            batch,
            heads,
            query_length,
            key_length,
        ) or not bool(torch.isfinite(probabilities).all().item()):
            raise Lfm2ComposedQkAttentionFoldError(
                "Cached parent attention probabilities differ from Q/K geometry."
            )
        if (
            bool((probabilities < 0.0).any().item())
            or bool((probabilities.masked_select(~mask).abs() > 0.0).any().item())
            or not torch.allclose(
                probabilities.sum(dim=-1),
                torch.ones(
                    (batch, heads, query_length),
                    dtype=torch.float32,
                ),
                rtol=2e-2,
                atol=2e-3,
            )
        ):
            raise Lfm2ComposedQkAttentionAuthenticationError(
                "Cached parent attention probabilities violate their causal "
                "normalization contract."
            )
        provenance = "authenticated_cached_parent_attention_probabilities"
    parent_value = case.parent_value.detach().to(
        device="cpu",
        dtype=torch.float32,
    )
    if (
        parent_value.ndim != 4
        or tuple(parent_value.shape[:2]) != (batch, int(key.shape[1]))
        or int(parent_value.shape[2]) != key_length
        or int(parent_value.shape[3]) != head_dim
    ):
        raise Lfm2ComposedQkAttentionFoldError(
            "Cached parent V differs from the authenticated GQA route."
        )
    repeated_value = parent_value.repeat_interleave(
        heads // int(parent_value.shape[1]),
        dim=1,
    )
    replayed_context = torch.matmul(probabilities, repeated_value)
    cached_context = case.parent_context.detach().to(device="cpu").contiguous()
    if cached_context.ndim != 4 or tuple(cached_context.shape) != tuple(
        replayed_context.shape
    ):
        raise Lfm2ComposedQkAttentionFoldError(
            "Cached parent context differs from the authenticated GQA geometry."
        )
    context_rtol = 2e-2
    context_atol = 2e-3
    replayed_for_comparison = replayed_context.to(torch.float32)
    cached_for_comparison = cached_context.to(torch.float32)
    difference = replayed_for_comparison.to(torch.float64) - cached_for_comparison.to(
        torch.float64
    )
    target_root_mean_square = float(
        cached_for_comparison.to(torch.float64).square().mean().sqrt().item()
    )
    relative_rmse = float(difference.square().mean().sqrt().item()) / max(
        target_root_mean_square,
        torch.finfo(torch.float64).tiny,
    )
    maximum_absolute_error = float(difference.abs().max().item())
    if not torch.allclose(
        replayed_for_comparison,
        cached_for_comparison,
        rtol=context_rtol,
        atol=context_atol,
    ):
        raise Lfm2ComposedQkAttentionAuthenticationError(
            "Cached parent Q/K/V, mask, scale, probabilities, and context do "
            "not replay within the authenticated trace-cache tolerance."
        )
    return (
        probabilities.to(torch.float64).contiguous(),
        mask.contiguous(),
        provenance,
        relative_rmse,
        maximum_absolute_error,
        context_rtol,
        context_atol,
    )


def _endpoint_case_from_cached_attention(
    case: Lfm2CachedAttentionFoldCase,
    *,
    trajectory_sha256: str,
    query_groups: Sequence[Sequence[int]],
    query_group_weights: Sequence[Sequence[float]],
) -> Lfm2EndpointQkResidualCase:
    torch = _torch()
    if not isinstance(case, Lfm2CachedAttentionFoldCase):
        raise TypeError("Endpoint adaptation requires cached attention evidence.")
    query_inputs = case.target_query_input.detach().to(device="cpu").contiguous()
    key_inputs = case.target_key_value_input.detach().to(device="cpu").contiguous()
    if query_inputs.dtype != torch.bfloat16 or key_inputs.dtype != torch.bfloat16:
        raise Lfm2ComposedQkAttentionFoldError(
            "Endpoint adaptation requires exact cached BF16 target inputs."
        )
    (
        probabilities,
        parent_mask,
        probability_provenance,
        context_replay_relative_rmse,
        context_replay_maximum_absolute_error,
        context_replay_rtol,
        context_replay_atol,
    ) = _cached_parent_attention_probabilities(case)
    batch, query_length, _hidden = map(int, query_inputs.shape)
    key_length = int(key_inputs.shape[1])
    head_dim = int(case.parent_query.shape[-1])

    def rope_term(value: Any | None, *, length: int, cosine: bool) -> Any:
        if value is None:
            fill = 1.0 if cosine else 0.0
            return torch.full(
                (batch, length, head_dim),
                fill,
                dtype=torch.bfloat16,
            )
        result = value.detach().to(device="cpu").contiguous()
        if result.dtype != torch.bfloat16 or tuple(result.shape) != (
            batch,
            length,
            head_dim,
        ):
            raise Lfm2ComposedQkAttentionFoldError(
                "Cached RoPE evidence differs from the parent route geometry."
            )
        return result

    endpoint = (
        case.parent_post_attention_residual_target_space.detach()
        .to(device="cpu", dtype=torch.float32)
        .contiguous()
        - case.target_pre_attention_residual.detach()
        .to(device="cpu", dtype=torch.float32)
        .contiguous()
    )
    row_weights = (
        torch.ones((batch, query_length), dtype=torch.bfloat16)
        if case.query_row_sqrt_weights is None
        else case.query_row_sqrt_weights.detach().to(device="cpu").contiguous()
    )
    return Lfm2EndpointQkResidualCase(
        target_query_inputs=query_inputs,
        target_key_value_inputs=key_inputs,
        parent_probabilities=probabilities,
        parent_causal_mask=parent_mask,
        parent_endpoint_target=endpoint,
        query_rope_cos=rope_term(
            case.query_cosine,
            length=query_length,
            cosine=True,
        ),
        query_rope_sin=rope_term(
            case.query_sine,
            length=query_length,
            cosine=False,
        ),
        key_rope_cos=rope_term(
            case.key_cosine,
            length=key_length,
            cosine=True,
        ),
        key_rope_sin=rope_term(
            case.key_sine,
            length=key_length,
            cosine=False,
        ),
        query_row_sqrt_weights=row_weights,
        query_groups=tuple(tuple(value) for value in query_groups),
        query_group_weights=tuple(
            tuple(float(weight) for weight in value) for value in query_group_weights
        ),
        trajectory_sha256=trajectory_sha256,
        parent_probability_provenance=probability_provenance,
        parent_context_replay_relative_rmse=context_replay_relative_rmse,
        parent_context_replay_maximum_absolute_error=(
            context_replay_maximum_absolute_error
        ),
        parent_context_replay_rtol=context_replay_rtol,
        parent_context_replay_atol=context_replay_atol,
    )


def _endpoint_cases_from_qk_bank(
    composed_qk: Lfm2EndpointQkStartingArtifact,
    bank: Lfm2CachedQkEvidenceBank,
    *,
    allow_legacy_recomputed_parent_probabilities: bool,
) -> tuple[Lfm2EndpointQkResidualCase, ...]:
    producer = composed_qk.producer_artifact
    cases = tuple(
        _endpoint_case_from_cached_attention(
            case,
            trajectory_sha256=trajectory,
            query_groups=producer.query_groups,
            query_group_weights=producer.query_barycentric_weights,
        )
        for case, trajectory in zip(
            bank.cases,
            bank.trajectory_sha256s,
            strict=True,
        )
    )
    if (
        any(
            case.parent_probability_provenance == "legacy_recomputed_parent_qk"
            for case in cases
        )
        and not allow_legacy_recomputed_parent_probabilities
    ):
        raise Lfm2ComposedQkAttentionAuthenticationError(
            "Production endpoint folding requires authenticated cached parent "
            "attention probabilities; legacy Q/K recomputation is disabled."
        )
    return cases


def _exact_qkv_producer(
    base: Lfm2CompiledStructuralAttentionProducerFold,
    *,
    query_weight: Any,
    key_weight: Any,
    value_weight: Any,
) -> Lfm2CompiledStructuralAttentionProducerFold:
    state = {
        "q_proj.weight": query_weight,
        "k_proj.weight": key_weight,
        "v_proj.weight": value_weight,
    }
    producer = replace(
        base,
        query_weight=query_weight.detach().to(device="cpu").contiguous(),
        key_weight=key_weight.detach().to(device="cpu").contiguous(),
        value_weight=value_weight.detach().to(device="cpu").contiguous(),
        source_derived_prior_sha256=_structural_state_sha256(state),
        compiled_state_sha256=_structural_state_sha256(state),
    )
    _verify_structural_producer(producer)
    return producer


def _qkvo_state(
    *,
    query_weight: Any,
    key_weight: Any,
    value_weight: Any,
    output_weight: Any,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "q_proj.weight": query_weight.detach().to(device="cpu").contiguous(),
            "k_proj.weight": key_weight.detach().to(device="cpu").contiguous(),
            "v_proj.weight": value_weight.detach().to(device="cpu").contiguous(),
            "out_proj.weight": output_weight.detach().to(device="cpu").contiguous(),
        }
    )


def _endpoint_metrics_nonregressing(
    candidate: Lfm2EndpointQkMetrics,
    baseline: Lfm2EndpointQkMetrics,
    *,
    tolerance: float,
) -> bool:
    return (
        candidate.endpoint_relative_rmse <= baseline.endpoint_relative_rmse + tolerance
        and candidate.probability_relative_rmse
        <= baseline.probability_relative_rmse + tolerance
        and candidate.js_divergence_mean <= baseline.js_divergence_mean + tolerance
        and candidate.js_divergence_worst <= baseline.js_divergence_worst + tolerance
        and candidate.top1_route_agreement_fraction + tolerance
        >= baseline.top1_route_agreement_fraction
        and candidate.maximum_causal_probability_leakage == 0.0
    )


@dataclass(frozen=True, slots=True)
class Lfm2CompiledComposedEndpointResidualBlock:
    """One authenticated FIT-only endpoint-QK then sequence-V/O block."""

    block_index: int
    fit_bank: Lfm2EndpointQkResidualEvidenceBank
    endpoint_seed: Lfm2EndpointQkvoSeed
    qk_proposal: Lfm2CompiledEndpointQkFitProposal
    residual_vo_seed: Lfm2SequenceEffectiveResidualVoSeed
    vo_artifact: Lfm2CompiledFixedQkSequenceEffectiveAttentionFold
    input_state_sha256: str
    output_state_sha256: str
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        index = self.block_index
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index
            not in {
                0,
                1,
            }
        ):
            raise Lfm2ComposedQkAttentionAuthenticationError(
                "Endpoint outer block index must be zero or one."
            )
        if (
            not isinstance(
                self.fit_bank,
                Lfm2EndpointQkResidualEvidenceBank,
            )
            or self.fit_bank.partition != "fit"
            or self.fit_bank.source_artifact_sha256
            != self.endpoint_seed.source_artifact_sha256
            or self.qk_proposal.seed is not self.endpoint_seed
            or self.qk_proposal.fit_bank is not self.fit_bank
            or self.qk_proposal.report.get("fit_evidence_sha256")
            != self.fit_bank.verify()
            or not self.qk_proposal.changed
            or not isinstance(
                self.residual_vo_seed,
                Lfm2SequenceEffectiveResidualVoSeed,
            )
            or not self.residual_vo_seed.verify_authentication()
            or not isinstance(
                self.vo_artifact,
                Lfm2CompiledFixedQkSequenceEffectiveAttentionFold,
            )
            or not self.vo_artifact.verify_authentication()
            or self.vo_artifact.residual_vo_seed is not self.residual_vo_seed
            or not _same_tensor(
                self.qk_proposal.query_weight,
                self.vo_artifact.producer_artifact.query_weight,
            )
            or not _same_tensor(
                self.qk_proposal.key_weight,
                self.vo_artifact.producer_artifact.key_weight,
            )
            or not _same_tensor(
                self.endpoint_seed.value_weight,
                self.residual_vo_seed.value_weight,
            )
            or not _same_tensor(
                self.endpoint_seed.output_weight,
                self.residual_vo_seed.output_weight,
            )
        ):
            raise Lfm2ComposedQkAttentionAuthenticationError(
                "Endpoint outer block did not preserve its exact FIT-only "
                "Q/K-to-V/O chain."
            )
        input_state = _qkvo_state(
            query_weight=self.endpoint_seed.query_weight,
            key_weight=self.endpoint_seed.key_weight,
            value_weight=self.endpoint_seed.value_weight,
            output_weight=self.endpoint_seed.output_weight,
        )
        input_sha = _required_sha256(
            self.input_state_sha256,
            "input_state_sha256",
        )
        output_sha = _required_sha256(
            self.output_state_sha256,
            "output_state_sha256",
        )
        if (
            input_sha != _structural_state_sha256(input_state)
            or output_sha != _structural_state_sha256(self.vo_artifact.state_dict())
            or self.residual_vo_seed.provenance.source_compiled_state_sha256
            != input_sha
            or self.residual_vo_seed.provenance.source_artifact_seal_sha256
            != self.endpoint_seed.source_artifact_sha256
        ):
            raise Lfm2ComposedQkAttentionAuthenticationError(
                "Endpoint outer block dense state chain is not authenticated."
            )
        qk_sha = self.qk_proposal.verify()
        vo_sha = _required_sha256(
            self.vo_artifact.report.get("report_sha256"),
            "vo_artifact.report_sha256",
        )
        body = dict(self.report)
        supplied = body.pop("block_sha256", None)
        expected = {
            "schema": LFM2_COMPOSED_ENDPOINT_RESIDUAL_BLOCK_SCHEMA,
            "block_index": index,
            "fit_evidence_sha256": self.fit_bank.evidence_sha256,
            "fit_trajectory_sha256s": list(self.fit_bank.trajectory_sha256s),
            "fit_target_receipt_sha256s": list(self.fit_bank.target_receipt_sha256s),
            "endpoint_seed_sha256": self.endpoint_seed.verify(),
            "source_inventory_sha256": (
                self.endpoint_seed.source_inventory.inventory_sha256
            ),
            "qk_fit_proposal_sha256": qk_sha,
            "residual_vo_seed_sha256": (self.residual_vo_seed.authentication_sha256),
            "vo_refit_report_sha256": vo_sha,
            "input_state_sha256": input_sha,
            "output_state_sha256": output_sha,
            "fit_check_accessible": False,
            "fit_check_cases_consumed": 0,
            "block_order": [
                "fixed_vo_endpoint_qk_fit",
                "fixed_qk_sequence_effective_vo_refit",
            ],
            "operation_audit": {name: 0 for name in _ZERO_OPERATION_FIELDS},
        }
        if supplied != _canonical_sha256(body) or any(
            body.get(name) != value for name, value in expected.items()
        ):
            raise Lfm2ComposedQkAttentionAuthenticationError(
                "Endpoint outer block receipt is invalid."
            )
        object.__setattr__(self, "input_state_sha256", input_sha)
        object.__setattr__(self, "output_state_sha256", output_sha)
        object.__setattr__(
            self,
            "report",
            MappingProxyType({**body, "block_sha256": supplied}),
        )

    @property
    def block_sha256(self) -> str:
        return str(self.report["block_sha256"])

    def state_dict(self) -> Mapping[str, Any]:
        return self.vo_artifact.state_dict()

    def verify(self) -> str:
        rebuilt = Lfm2CompiledComposedEndpointResidualBlock(
            block_index=self.block_index,
            fit_bank=self.fit_bank,
            endpoint_seed=self.endpoint_seed,
            qk_proposal=self.qk_proposal,
            residual_vo_seed=self.residual_vo_seed,
            vo_artifact=self.vo_artifact,
            input_state_sha256=self.input_state_sha256,
            output_state_sha256=self.output_state_sha256,
            report=self.report,
        )
        return rebuilt.block_sha256


@dataclass(frozen=True, slots=True)
class Lfm2CompiledComposedEndpointResidualAttentionFold:
    """Authenticated fifth-arm dense QKVO artifact with a development gate."""

    policy: Lfm2ComposedEndpointResidualAttentionFoldPolicy
    composed_qk: Lfm2EndpointQkStartingArtifact
    fit_value_output_artifact: Lfm2CompiledStructuralAttentionValueOutputFold
    source_inventory: Lfm2DenseQkvoSourceInventory
    blocks: tuple[Lfm2CompiledComposedEndpointResidualBlock, ...]
    evaluation_fit_bank: Lfm2EndpointQkResidualEvidenceBank
    evaluation_fit_check_bank: Lfm2EndpointQkResidualEvidenceBank
    baseline_evaluation_seed: Lfm2EndpointQkvoSeed
    final_evaluation_seed: Lfm2EndpointQkvoSeed
    baseline_fit_metrics: Lfm2EndpointQkMetrics
    final_fit_metrics: Lfm2EndpointQkMetrics
    baseline_fit_check_metrics: Lfm2EndpointQkMetrics
    final_fit_check_metrics: Lfm2EndpointQkMetrics
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            not isinstance(
                self.policy,
                Lfm2ComposedEndpointResidualAttentionFoldPolicy,
            )
            or not self.policy.verify_authentication()
        ):
            raise Lfm2ComposedQkAttentionAuthenticationError(
                "Fifth-arm policy authentication failed."
            )
        composed_sha = self.composed_qk.verify_authentication()
        runtime = self.composed_qk.primitive_fold.runtime
        fit_cases = tuple(self.composed_qk.qk_fit_bank.cases)
        vo_sha, _vo_receipts = _verify_value_output_artifact(
            self.fit_value_output_artifact,
            composed=self.composed_qk,
            cases=fit_cases,
            query_norm_weight=runtime.target_query_norm_weight,
            key_norm_weight=runtime.target_key_norm_weight,
            norm_epsilon=runtime.norm_epsilon,
            actual_child_contexts=None,
        )
        if (
            not isinstance(
                self.source_inventory,
                Lfm2DenseQkvoSourceInventory,
            )
            or not self.source_inventory.verify()
            or self.source_inventory.structural_source_state_sha256
            != self.composed_qk.producer_artifact.source_state_sha256_before
            or self.source_inventory.logical_parameters
            != (
                self.composed_qk.producer_artifact.source_logical_parameters
                + self.composed_qk.topology.source_hidden_size
                * self.composed_qk.topology.source_query_heads
                * self.composed_qk.topology.head_dim
            )
        ):
            raise Lfm2ComposedQkAttentionAuthenticationError(
                "Fifth-arm source Q/K/V/O inventory authentication failed."
            )
        blocks = tuple(self.blocks)
        if (
            not blocks
            or len(blocks) > self.policy.maximum_outer_blocks
            or len(blocks) > 2
            or any(
                not isinstance(
                    block,
                    Lfm2CompiledComposedEndpointResidualBlock,
                )
                for block in blocks
            )
            or tuple(block.block_index for block in blocks) != tuple(range(len(blocks)))
            or any(
                block.endpoint_seed.source_inventory.inventory_sha256
                != self.source_inventory.inventory_sha256
                for block in blocks
            )
        ):
            raise Lfm2ComposedQkAttentionAuthenticationError(
                "Fifth-arm outer blocks are unbounded, reordered, or alien."
            )
        initial_state = _qkvo_state(
            query_weight=self.composed_qk.producer_artifact.query_weight,
            key_weight=self.composed_qk.producer_artifact.key_weight,
            value_weight=self.fit_value_output_artifact.value_weight,
            output_weight=self.fit_value_output_artifact.output_weight,
        )
        previous_sha = _structural_state_sha256(initial_state)
        for block in blocks:
            block.verify()
            if block.input_state_sha256 != previous_sha:
                raise Lfm2ComposedQkAttentionAuthenticationError(
                    "Fifth-arm outer block states are not one ordered chain."
                )
            previous_sha = block.output_state_sha256
        final_state = blocks[-1].state_dict()
        if previous_sha != _structural_state_sha256(final_state):
            raise Lfm2ComposedQkAttentionAuthenticationError(
                "Fifth-arm final dense Q/K/V/O state is unauthenticated."
            )
        expected_evaluation_artifact = _canonical_sha256(
            {
                "schema": "neural-forge-lfm2-endpoint-final-evaluation-v1",
                **_endpoint_qk_start_binding(self.composed_qk, composed_sha),
                "block_sha256s": [block.block_sha256 for block in blocks],
                "initial_state_sha256": _structural_state_sha256(initial_state),
                "final_state_sha256": previous_sha,
            }
        )
        fit_bank_sha = self.evaluation_fit_bank.verify()
        check_bank_sha = self.evaluation_fit_check_bank.verify()
        reconstructed_fit_bank = seal_lfm2_endpoint_qk_fit_bank(
            _endpoint_cases_from_qk_bank(
                self.composed_qk,
                self.composed_qk.qk_fit_bank,
                allow_legacy_recomputed_parent_probabilities=(
                    self.policy.allow_legacy_recomputed_parent_probabilities
                ),
            ),
            source_split_sha256=(self.composed_qk.qk_fit_bank.source_split_sha256),
            source_artifact_sha256=expected_evaluation_artifact,
        )
        reconstructed_fit_check_bank = seal_lfm2_endpoint_qk_fit_check_bank(
            _endpoint_cases_from_qk_bank(
                self.composed_qk,
                self.composed_qk.qk_fit_check_bank,
                allow_legacy_recomputed_parent_probabilities=(
                    self.policy.allow_legacy_recomputed_parent_probabilities
                ),
            ),
            source_split_sha256=(
                self.composed_qk.qk_fit_check_bank.source_split_sha256
            ),
            source_artifact_sha256=expected_evaluation_artifact,
        )
        if (
            self.evaluation_fit_bank.partition != "fit"
            or self.evaluation_fit_check_bank.partition != "fit_check"
            or fit_bank_sha != reconstructed_fit_bank.evidence_sha256
            or check_bank_sha != reconstructed_fit_check_bank.evidence_sha256
            or self.evaluation_fit_bank.source_artifact_sha256
            != self.evaluation_fit_check_bank.source_artifact_sha256
            or self.evaluation_fit_bank.source_artifact_sha256
            != expected_evaluation_artifact
            or self.baseline_evaluation_seed.source_artifact_sha256
            != self.evaluation_fit_bank.source_artifact_sha256
            or self.final_evaluation_seed.source_artifact_sha256
            != self.evaluation_fit_bank.source_artifact_sha256
            or self.baseline_evaluation_seed.source_inventory.inventory_sha256
            != self.source_inventory.inventory_sha256
            or self.final_evaluation_seed.source_inventory.inventory_sha256
            != self.source_inventory.inventory_sha256
            or tuple(self.evaluation_fit_bank.trajectory_sha256s)
            != tuple(self.composed_qk.qk_fit_bank.trajectory_sha256s)
            or tuple(self.evaluation_fit_check_bank.trajectory_sha256s)
            != tuple(self.composed_qk.qk_fit_check_bank.trajectory_sha256s)
        ):
            raise Lfm2ComposedQkAttentionAuthenticationError(
                "Final endpoint evaluation banks/seeds are not the exact "
                "ordered FIT/FIT-CHECK evidence."
            )
        baseline_state = _qkvo_state(
            query_weight=self.baseline_evaluation_seed.query_weight,
            key_weight=self.baseline_evaluation_seed.key_weight,
            value_weight=self.baseline_evaluation_seed.value_weight,
            output_weight=self.baseline_evaluation_seed.output_weight,
        )
        final_evaluation_state = _qkvo_state(
            query_weight=self.final_evaluation_seed.query_weight,
            key_weight=self.final_evaluation_seed.key_weight,
            value_weight=self.final_evaluation_seed.value_weight,
            output_weight=self.final_evaluation_seed.output_weight,
        )
        devices = self.policy.endpoint_policy.compute_devices
        recomputed_baseline_fit = evaluate_lfm2_endpoint_qkvo_bank(
            reconstructed_fit_bank,
            self.baseline_evaluation_seed,
            compute_devices=devices,
        )
        recomputed_final_fit = evaluate_lfm2_endpoint_qkvo_bank(
            reconstructed_fit_bank,
            self.final_evaluation_seed,
            compute_devices=devices,
        )
        recomputed_baseline_check = evaluate_lfm2_endpoint_qkvo_bank(
            reconstructed_fit_check_bank,
            self.baseline_evaluation_seed,
            compute_devices=devices,
        )
        recomputed_final_check = evaluate_lfm2_endpoint_qkvo_bank(
            reconstructed_fit_check_bank,
            self.final_evaluation_seed,
            compute_devices=devices,
        )
        tolerance = self.policy.endpoint_policy.metric_regression_tolerance
        if (
            _structural_state_sha256(baseline_state)
            != _structural_state_sha256(initial_state)
            or _structural_state_sha256(final_evaluation_state)
            != _structural_state_sha256(final_state)
            or self.baseline_fit_metrics.as_dict() != recomputed_baseline_fit.as_dict()
            or self.final_fit_metrics.as_dict() != recomputed_final_fit.as_dict()
            or self.baseline_fit_check_metrics.as_dict()
            != recomputed_baseline_check.as_dict()
            or self.final_fit_check_metrics.as_dict()
            != recomputed_final_check.as_dict()
            or self.final_fit_metrics.endpoint_relative_rmse
            > self.baseline_fit_metrics.endpoint_relative_rmse
            - self.policy.endpoint_policy.minimum_fit_endpoint_improvement
            or not _endpoint_metrics_nonregressing(
                self.final_fit_check_metrics,
                self.baseline_fit_check_metrics,
                tolerance=tolerance,
            )
        ):
            raise Lfm2ComposedQkAttentionAuthenticationError(
                "Fifth-arm metrics do not replay, or final FIT improvement/"
                "development-gate non-regression failed."
            )
        block_shas = [block.block_sha256 for block in blocks]
        body = dict(self.report)
        seal = body.pop("artifact_sha256", None)
        target_parameters = sum(int(value.numel()) for value in final_state.values())
        evaluation_cases = (
            *self.evaluation_fit_bank.cases,
            *self.evaluation_fit_check_bank.cases,
        )
        probability_modes = sorted(
            {case.parent_probability_provenance for case in evaluation_cases}
        )
        cached_probability_case_count = sum(
            case.parent_probability_provenance
            == "authenticated_cached_parent_attention_probabilities"
            for case in evaluation_cases
        )
        legacy_probability_case_count = sum(
            case.parent_probability_provenance == "legacy_recomputed_parent_qk"
            for case in evaluation_cases
        )
        expected = {
            "schema": LFM2_COMPOSED_ENDPOINT_RESIDUAL_SCHEMA,
            "candidate_id": ENDPOINT_RESIDUAL_ALTERNATING_QKVO_CANDIDATE_ID,
            "accepted": True,
            "policy": self.policy.as_dict(),
            "policy_sha256": self.policy.authentication_sha256,
            **_endpoint_qk_start_binding(self.composed_qk, composed_sha),
            "fit_value_output_artifact_sha256": vo_sha,
            "source_inventory": self.source_inventory.as_dict(),
            "source_inventory_sha256": self.source_inventory.inventory_sha256,
            "source_structural_state_sha256": (
                self.composed_qk.producer_artifact.source_state_sha256_before
            ),
            "evaluation_fit_evidence_sha256": fit_bank_sha,
            "evaluation_fit_check_evidence_sha256": check_bank_sha,
            "fit_trajectory_sha256s": list(self.evaluation_fit_bank.trajectory_sha256s),
            "fit_check_trajectory_sha256s": list(
                self.evaluation_fit_check_bank.trajectory_sha256s
            ),
            "fit_target_receipt_sha256s": list(
                self.evaluation_fit_bank.target_receipt_sha256s
            ),
            "fit_check_target_receipt_sha256s": list(
                self.evaluation_fit_check_bank.target_receipt_sha256s
            ),
            "block_sha256s": block_shas,
            "outer_block_count": len(blocks),
            "initial_state_sha256": _structural_state_sha256(initial_state),
            "final_state_sha256": _structural_state_sha256(final_state),
            "baseline_evaluation_seed_sha256": (self.baseline_evaluation_seed.verify()),
            "final_evaluation_seed_sha256": self.final_evaluation_seed.verify(),
            "baseline_fit_metrics": self.baseline_fit_metrics.as_dict(),
            "final_fit_metrics": self.final_fit_metrics.as_dict(),
            "baseline_fit_check_metrics": (self.baseline_fit_check_metrics.as_dict()),
            "final_fit_check_metrics": self.final_fit_check_metrics.as_dict(),
            "fit_check_endpoint_and_route_nonregressing": True,
            "fit_check_first_metric_evaluation_after_all_outer_blocks": True,
            "fit_check_triggered_refit": False,
            "fit_check_role": (
                "reused_development_acceptance_gate_not_unbiased_validation"
            ),
            "unbiased_generalization_claimed": False,
            "fresh_whole_child_intelligence_validation_required": True,
            "proof_scope": (
                "source_bound_local_attention_operator_fold_not_whole_child_intelligence"
            ),
            "direct_parent_probability_barycenter_target": True,
            "direct_parent_attention_endpoint_target": True,
            "parent_probability_evidence_modes": probability_modes,
            "authenticated_cached_parent_probability_case_count": (
                cached_probability_case_count
            ),
            "legacy_recomputed_parent_probability_case_count": (
                legacy_probability_case_count
            ),
            "all_parent_probability_targets_from_authenticated_cache": (
                legacy_probability_case_count == 0
            ),
            "legacy_parent_probability_recomputation_allowed": (
                self.policy.allow_legacy_recomputed_parent_probabilities
            ),
            "ordinary_dense_target_qkvo": True,
            "source_logical_parameters": self.source_inventory.logical_parameters,
            "target_logical_parameters": target_parameters,
            "parameter_compaction_ratio": (
                self.source_inventory.logical_parameters / target_parameters
            ),
            "operation_audit": {name: 0 for name in _ZERO_OPERATION_FIELDS},
        }
        if seal != _canonical_sha256(body) or any(
            body.get(name) != value for name, value in expected.items()
        ):
            raise Lfm2ComposedQkAttentionAuthenticationError(
                "Fifth-arm endpoint-residual artifact receipt is invalid."
            )
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(
            self,
            "report",
            MappingProxyType({**body, "artifact_sha256": seal}),
        )

    @property
    def candidate_id(self) -> str:
        return ENDPOINT_RESIDUAL_ALTERNATING_QKVO_CANDIDATE_ID

    @property
    def accepted(self) -> bool:
        return True

    @property
    def artifact_sha256(self) -> str:
        return str(self.report["artifact_sha256"])

    @property
    def producer_artifact(self) -> Lfm2CompiledStructuralAttentionProducerFold:
        return self.blocks[-1].vo_artifact.producer_artifact

    @property
    def topology(self) -> Any:
        return self.producer_artifact.topology

    @property
    def compiled_state_sha256(self) -> str:
        return _structural_state_sha256(self.state_dict())

    def state_dict(self) -> Mapping[str, Any]:
        return self.blocks[-1].state_dict()

    def verify_authentication(self) -> str:
        rebuilt = Lfm2CompiledComposedEndpointResidualAttentionFold(
            policy=self.policy,
            composed_qk=self.composed_qk,
            fit_value_output_artifact=self.fit_value_output_artifact,
            source_inventory=self.source_inventory,
            blocks=self.blocks,
            evaluation_fit_bank=self.evaluation_fit_bank,
            evaluation_fit_check_bank=self.evaluation_fit_check_bank,
            baseline_evaluation_seed=self.baseline_evaluation_seed,
            final_evaluation_seed=self.final_evaluation_seed,
            baseline_fit_metrics=self.baseline_fit_metrics,
            final_fit_metrics=self.final_fit_metrics,
            baseline_fit_check_metrics=self.baseline_fit_check_metrics,
            final_fit_check_metrics=self.final_fit_check_metrics,
            report=self.report,
        )
        return rebuilt.artifact_sha256

    def as_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dict(self.report)))


def compile_lfm2_composed_endpoint_residual_attention_fold(
    *,
    composed_qk: Lfm2EndpointQkStartingArtifact,
    fit_value_output_artifact: Lfm2CompiledStructuralAttentionValueOutputFold,
    source_query_weight: Any,
    source_key_weight: Any,
    source_value_weight: Any,
    source_output_weight: Any,
    target_query_weight_template: Any,
    target_key_weight_template: Any,
    target_value_weight_template: Any,
    target_output_weight_template: Any,
    query_norm_weight: Any,
    key_norm_weight: Any,
    norm_epsilon: float = 1e-5,
    policy: Lfm2ComposedEndpointResidualAttentionFoldPolicy | None = None,
) -> Lfm2CompiledComposedEndpointResidualAttentionFold:
    """Compile the fifth arm using at most two FIT-only QK→VO blocks."""

    _verify_endpoint_qk_start(composed_qk)
    selected_policy = policy or Lfm2ComposedEndpointResidualAttentionFoldPolicy()
    if (
        not isinstance(
            selected_policy,
            Lfm2ComposedEndpointResidualAttentionFoldPolicy,
        )
        or not selected_policy.verify_authentication()
    ):
        raise TypeError("policy must be an authenticated composed endpoint policy.")
    composed_sha = composed_qk.verify_authentication()
    query_norm, key_norm, epsilon = _stage_two_runtime_contract(
        composed_qk,
        query_norm_weight=query_norm_weight,
        key_norm_weight=key_norm_weight,
        norm_epsilon=norm_epsilon,
    )
    source_values = {
        "q_proj.weight": source_query_weight,
        "k_proj.weight": source_key_weight,
        "v_proj.weight": source_value_weight,
        "out_proj.weight": source_output_weight,
    }
    source_structural_sha = _structural_state_sha256(source_values)
    producer = composed_qk.producer_artifact
    if (
        source_structural_sha != producer.source_state_sha256_before
        or source_structural_sha != producer.source_state_sha256_after
    ):
        raise Lfm2ComposedQkAttentionAuthenticationError(
            "Actual source Q/K/V/O tensors differ from the composed producer's "
            "authenticated source inventory."
        )
    source_inventory = seal_lfm2_dense_qkvo_source_inventory(
        source_query_weight=source_query_weight,
        source_key_weight=source_key_weight,
        source_value_weight=source_value_weight,
        source_output_weight=source_output_weight,
    )
    fit_cases = tuple(composed_qk.qk_fit_bank.cases)
    initial_vo_sha, _initial_vo_receipts = _verify_value_output_artifact(
        fit_value_output_artifact,
        composed=composed_qk,
        cases=fit_cases,
        query_norm_weight=query_norm,
        key_norm_weight=key_norm,
        norm_epsilon=epsilon,
        actual_child_contexts=None,
    )
    fit_endpoint_cases = _endpoint_cases_from_qk_bank(
        composed_qk,
        composed_qk.qk_fit_bank,
        allow_legacy_recomputed_parent_probabilities=(
            selected_policy.allow_legacy_recomputed_parent_probabilities
        ),
    )
    current_query = producer.query_weight
    current_key = producer.key_weight
    current_value = fit_value_output_artifact.value_weight
    current_output = fit_value_output_artifact.output_weight
    initial_state = _qkvo_state(
        query_weight=current_query,
        key_weight=current_key,
        value_weight=current_value,
        output_weight=current_output,
    )
    previous_artifact_sha = _canonical_sha256(
        {
            "schema": "neural-forge-lfm2-endpoint-block-input-v1",
            **_endpoint_qk_start_binding(composed_qk, composed_sha),
            "fit_value_output_artifact_sha256": initial_vo_sha,
            "dense_state_sha256": _structural_state_sha256(initial_state),
        }
    )
    blocks: list[Lfm2CompiledComposedEndpointResidualBlock] = []
    for block_index in range(selected_policy.maximum_outer_blocks):
        input_state = _qkvo_state(
            query_weight=current_query,
            key_weight=current_key,
            value_weight=current_value,
            output_weight=current_output,
        )
        input_state_sha = _structural_state_sha256(input_state)
        block_source_artifact = _canonical_sha256(
            {
                "schema": "neural-forge-lfm2-endpoint-block-input-v1",
                "block_index": block_index,
                "previous_artifact_sha256": previous_artifact_sha,
                "input_state_sha256": input_state_sha,
                "source_inventory_sha256": source_inventory.inventory_sha256,
            }
        )
        fit_bank = seal_lfm2_endpoint_qk_fit_bank(
            fit_endpoint_cases,
            source_split_sha256=composed_qk.qk_fit_bank.source_split_sha256,
            source_artifact_sha256=block_source_artifact,
        )
        endpoint_seed = build_lfm2_endpoint_qkvo_seed(
            query_weight=current_query,
            key_weight=current_key,
            value_weight=current_value,
            output_weight=current_output,
            query_norm_weight=query_norm,
            key_norm_weight=key_norm,
            norm_epsilon=epsilon,
            source_inventory=source_inventory,
            source_artifact_sha256=block_source_artifact,
            target_query_weight_template=target_query_weight_template,
            target_key_weight_template=target_key_weight_template,
            target_value_weight_template=target_value_weight_template,
            target_output_weight_template=target_output_weight_template,
        )
        qk_proposal = compile_lfm2_cached_endpoint_qk_fit_proposal(
            seed=endpoint_seed,
            fit_bank=fit_bank,
            policy=selected_policy.endpoint_policy,
        )
        if not qk_proposal.changed:
            break
        rebound_producer = _exact_qkv_producer(
            producer,
            query_weight=qk_proposal.query_weight,
            key_weight=qk_proposal.key_weight,
            value_weight=current_value,
        )
        provenance = Lfm2SequenceEffectiveResidualVoSeedProvenance(
            source_artifact_schema=(
                "neural-forge-lfm2-endpoint-qk-fixed-vo-block-input-v1"
            ),
            source_artifact_seal_sha256=block_source_artifact,
            source_compiled_state_sha256=input_state_sha,
            source_value_weight_sha256=_structural_tensor_sha256(current_value),
            source_output_weight_sha256=_structural_tensor_sha256(current_output),
            evidence_split="fit",
        )
        residual_vo_seed = Lfm2SequenceEffectiveResidualVoSeed(
            value_weight=current_value,
            output_weight=current_output,
            provenance=provenance,
        )
        vo_artifact = compile_lfm2_fixed_qk_sequence_effective_attention_fold(
            producer=rebound_producer,
            target_value_weight_template=target_value_weight_template,
            target_output_weight_template=target_output_weight_template,
            fit_cases=fit_cases,
            query_norm_weight=query_norm,
            key_norm_weight=key_norm,
            norm_epsilon=epsilon,
            policy=selected_policy.sequence_policy,
            residual_vo_seed=residual_vo_seed,
        )
        if (
            not vo_artifact.verify_authentication()
            or not _same_tensor(
                qk_proposal.query_weight,
                vo_artifact.producer_artifact.query_weight,
            )
            or not _same_tensor(
                qk_proposal.key_weight,
                vo_artifact.producer_artifact.key_weight,
            )
        ):
            raise Lfm2ComposedQkAttentionAuthenticationError(
                "Fifth-arm V/O refit changed the endpoint-corrected Q/K state."
            )
        output_state_sha = _structural_state_sha256(vo_artifact.state_dict())
        vo_report_sha = _required_sha256(
            vo_artifact.report.get("report_sha256"),
            "vo_artifact.report_sha256",
        )
        block_body = {
            "schema": LFM2_COMPOSED_ENDPOINT_RESIDUAL_BLOCK_SCHEMA,
            "block_index": block_index,
            "fit_evidence_sha256": fit_bank.evidence_sha256,
            "fit_trajectory_sha256s": list(fit_bank.trajectory_sha256s),
            "fit_target_receipt_sha256s": list(fit_bank.target_receipt_sha256s),
            "endpoint_seed_sha256": endpoint_seed.seed_sha256,
            "source_inventory_sha256": source_inventory.inventory_sha256,
            "qk_fit_proposal_sha256": qk_proposal.report_sha256,
            "residual_vo_seed_sha256": (residual_vo_seed.authentication_sha256),
            "vo_refit_report_sha256": vo_report_sha,
            "input_state_sha256": input_state_sha,
            "output_state_sha256": output_state_sha,
            "fit_check_accessible": False,
            "fit_check_cases_consumed": 0,
            "block_order": [
                "fixed_vo_endpoint_qk_fit",
                "fixed_qk_sequence_effective_vo_refit",
            ],
            "operation_audit": {name: 0 for name in _ZERO_OPERATION_FIELDS},
        }
        block = Lfm2CompiledComposedEndpointResidualBlock(
            block_index=block_index,
            fit_bank=fit_bank,
            endpoint_seed=endpoint_seed,
            qk_proposal=qk_proposal,
            residual_vo_seed=residual_vo_seed,
            vo_artifact=vo_artifact,
            input_state_sha256=input_state_sha,
            output_state_sha256=output_state_sha,
            report={
                **block_body,
                "block_sha256": _canonical_sha256(block_body),
            },
        )
        blocks.append(block)
        current_query = vo_artifact.producer_artifact.query_weight
        current_key = vo_artifact.producer_artifact.key_weight
        current_value = vo_artifact.value_weight
        current_output = vo_artifact.output_weight
        previous_artifact_sha = block.block_sha256
    if not blocks:
        raise Lfm2ComposedQkAttentionFoldError(
            "The fifth arm found no exact-BF16 endpoint Q/K improvement on FIT."
        )

    # All Q/K and V/O updates are frozen above.  Only now is FIT-CHECK
    # transformed or numerically evaluated.
    fit_check_endpoint_cases = _endpoint_cases_from_qk_bank(
        composed_qk,
        composed_qk.qk_fit_check_bank,
        allow_legacy_recomputed_parent_probabilities=(
            selected_policy.allow_legacy_recomputed_parent_probabilities
        ),
    )
    evaluation_artifact = _canonical_sha256(
        {
            "schema": "neural-forge-lfm2-endpoint-final-evaluation-v1",
            **_endpoint_qk_start_binding(composed_qk, composed_sha),
            "block_sha256s": [block.block_sha256 for block in blocks],
            "initial_state_sha256": _structural_state_sha256(initial_state),
            "final_state_sha256": blocks[-1].output_state_sha256,
        }
    )
    evaluation_fit_bank = seal_lfm2_endpoint_qk_fit_bank(
        fit_endpoint_cases,
        source_split_sha256=composed_qk.qk_fit_bank.source_split_sha256,
        source_artifact_sha256=evaluation_artifact,
    )
    evaluation_fit_check_bank = seal_lfm2_endpoint_qk_fit_check_bank(
        fit_check_endpoint_cases,
        source_split_sha256=composed_qk.qk_fit_check_bank.source_split_sha256,
        source_artifact_sha256=evaluation_artifact,
    )

    def evaluation_seed(
        query_weight: Any,
        key_weight: Any,
        value_weight: Any,
        output_weight: Any,
    ) -> Lfm2EndpointQkvoSeed:
        return build_lfm2_endpoint_qkvo_seed(
            query_weight=query_weight,
            key_weight=key_weight,
            value_weight=value_weight,
            output_weight=output_weight,
            query_norm_weight=query_norm,
            key_norm_weight=key_norm,
            norm_epsilon=epsilon,
            source_inventory=source_inventory,
            source_artifact_sha256=evaluation_artifact,
            target_query_weight_template=target_query_weight_template,
            target_key_weight_template=target_key_weight_template,
            target_value_weight_template=target_value_weight_template,
            target_output_weight_template=target_output_weight_template,
        )

    baseline_seed = evaluation_seed(
        producer.query_weight,
        producer.key_weight,
        fit_value_output_artifact.value_weight,
        fit_value_output_artifact.output_weight,
    )
    final_seed = evaluation_seed(
        current_query,
        current_key,
        current_value,
        current_output,
    )
    devices = selected_policy.endpoint_policy.compute_devices
    baseline_fit_metrics = evaluate_lfm2_endpoint_qkvo_bank(
        evaluation_fit_bank,
        baseline_seed,
        compute_devices=devices,
    )
    final_fit_metrics = evaluate_lfm2_endpoint_qkvo_bank(
        evaluation_fit_bank,
        final_seed,
        compute_devices=devices,
    )
    baseline_check_metrics = evaluate_lfm2_endpoint_qkvo_bank(
        evaluation_fit_check_bank,
        baseline_seed,
        compute_devices=devices,
    )
    final_check_metrics = evaluate_lfm2_endpoint_qkvo_bank(
        evaluation_fit_check_bank,
        final_seed,
        compute_devices=devices,
    )
    tolerance = selected_policy.endpoint_policy.metric_regression_tolerance
    if (
        final_fit_metrics.endpoint_relative_rmse
        > baseline_fit_metrics.endpoint_relative_rmse
        - selected_policy.endpoint_policy.minimum_fit_endpoint_improvement
        or not _endpoint_metrics_nonregressing(
            final_check_metrics,
            baseline_check_metrics,
            tolerance=tolerance,
        )
    ):
        raise Lfm2ComposedQkAttentionFoldError(
            "The fifth arm failed final FIT improvement or development-gate "
            f"endpoint/route non-regression; FIT baseline={baseline_fit_metrics.as_dict()!r}; "
            f"FIT final={final_fit_metrics.as_dict()!r}; FIT-CHECK baseline="
            f"{baseline_check_metrics.as_dict()!r}; FIT-CHECK final="
            f"{final_check_metrics.as_dict()!r}."
        )
    final_state = blocks[-1].state_dict()
    target_parameters = sum(int(value.numel()) for value in final_state.values())
    evaluation_cases = (
        *evaluation_fit_bank.cases,
        *evaluation_fit_check_bank.cases,
    )
    probability_modes = sorted(
        {case.parent_probability_provenance for case in evaluation_cases}
    )
    cached_probability_case_count = sum(
        case.parent_probability_provenance
        == "authenticated_cached_parent_attention_probabilities"
        for case in evaluation_cases
    )
    legacy_probability_case_count = sum(
        case.parent_probability_provenance == "legacy_recomputed_parent_qk"
        for case in evaluation_cases
    )
    body = {
        "schema": LFM2_COMPOSED_ENDPOINT_RESIDUAL_SCHEMA,
        "candidate_id": ENDPOINT_RESIDUAL_ALTERNATING_QKVO_CANDIDATE_ID,
        "accepted": True,
        "policy": selected_policy.as_dict(),
        "policy_sha256": selected_policy.authentication_sha256,
        **_endpoint_qk_start_binding(composed_qk, composed_sha),
        "fit_value_output_artifact_sha256": initial_vo_sha,
        "source_inventory": source_inventory.as_dict(),
        "source_inventory_sha256": source_inventory.inventory_sha256,
        "source_structural_state_sha256": source_structural_sha,
        "evaluation_fit_evidence_sha256": (evaluation_fit_bank.evidence_sha256),
        "evaluation_fit_check_evidence_sha256": (
            evaluation_fit_check_bank.evidence_sha256
        ),
        "fit_trajectory_sha256s": list(evaluation_fit_bank.trajectory_sha256s),
        "fit_check_trajectory_sha256s": list(
            evaluation_fit_check_bank.trajectory_sha256s
        ),
        "fit_target_receipt_sha256s": list(evaluation_fit_bank.target_receipt_sha256s),
        "fit_check_target_receipt_sha256s": list(
            evaluation_fit_check_bank.target_receipt_sha256s
        ),
        "block_sha256s": [block.block_sha256 for block in blocks],
        "outer_block_count": len(blocks),
        "initial_state_sha256": _structural_state_sha256(initial_state),
        "final_state_sha256": _structural_state_sha256(final_state),
        "baseline_evaluation_seed_sha256": baseline_seed.seed_sha256,
        "final_evaluation_seed_sha256": final_seed.seed_sha256,
        "baseline_fit_metrics": baseline_fit_metrics.as_dict(),
        "final_fit_metrics": final_fit_metrics.as_dict(),
        "baseline_fit_check_metrics": baseline_check_metrics.as_dict(),
        "final_fit_check_metrics": final_check_metrics.as_dict(),
        "fit_check_endpoint_and_route_nonregressing": True,
        "fit_check_first_metric_evaluation_after_all_outer_blocks": True,
        "fit_check_triggered_refit": False,
        "fit_check_role": (
            "reused_development_acceptance_gate_not_unbiased_validation"
        ),
        "unbiased_generalization_claimed": False,
        "fresh_whole_child_intelligence_validation_required": True,
        "proof_scope": (
            "source_bound_local_attention_operator_fold_not_whole_child_intelligence"
        ),
        "direct_parent_probability_barycenter_target": True,
        "direct_parent_attention_endpoint_target": True,
        "parent_probability_evidence_modes": probability_modes,
        "authenticated_cached_parent_probability_case_count": (
            cached_probability_case_count
        ),
        "legacy_recomputed_parent_probability_case_count": (
            legacy_probability_case_count
        ),
        "all_parent_probability_targets_from_authenticated_cache": (
            legacy_probability_case_count == 0
        ),
        "legacy_parent_probability_recomputation_allowed": (
            selected_policy.allow_legacy_recomputed_parent_probabilities
        ),
        "ordinary_dense_target_qkvo": True,
        "source_logical_parameters": source_inventory.logical_parameters,
        "target_logical_parameters": target_parameters,
        "parameter_compaction_ratio": (
            source_inventory.logical_parameters / target_parameters
        ),
        "operation_audit": {name: 0 for name in _ZERO_OPERATION_FIELDS},
    }
    return Lfm2CompiledComposedEndpointResidualAttentionFold(
        policy=selected_policy,
        composed_qk=composed_qk,
        fit_value_output_artifact=fit_value_output_artifact,
        source_inventory=source_inventory,
        blocks=tuple(blocks),
        evaluation_fit_bank=evaluation_fit_bank,
        evaluation_fit_check_bank=evaluation_fit_check_bank,
        baseline_evaluation_seed=baseline_seed,
        final_evaluation_seed=final_seed,
        baseline_fit_metrics=baseline_fit_metrics,
        final_fit_metrics=final_fit_metrics,
        baseline_fit_check_metrics=baseline_check_metrics,
        final_fit_check_metrics=final_check_metrics,
        report={**body, "artifact_sha256": _canonical_sha256(body)},
    )


__all__ = [
    "ENDPOINT_RESIDUAL_ALTERNATING_QKVO_CANDIDATE_ID",
    "LFM2_COMPOSED_ENDPOINT_RESIDUAL_BLOCK_SCHEMA",
    "LFM2_COMPOSED_ENDPOINT_RESIDUAL_POLICY_SCHEMA",
    "LFM2_COMPOSED_ENDPOINT_RESIDUAL_SCHEMA",
    "LFM2_COMPOSED_QK_PRODUCER_SCHEMA",
    "LFM2_COMPOSED_QK_STAGE_ONE_ATTEMPT_SCHEMA",
    "LFM2_COMPOSED_QK_SEQUENCE_EFFECTIVE_SCHEMA",
    "LFM2_ENDPOINT_QK_START_SCHEMA",
    "SCORE_ROUTED_RESIDUAL_SEEDED_SEQUENCE_EFFECTIVE_CANDIDATE_ID",
    "Lfm2CompiledComposedEndpointResidualAttentionFold",
    "Lfm2CompiledComposedEndpointResidualBlock",
    "Lfm2CompiledComposedQkAttentionProducerFold",
    "Lfm2CompiledComposedQkSequenceEffectiveAttentionFold",
    "Lfm2AuthenticatedEndpointQkStart",
    "Lfm2ComposedQkStageOneAttempt",
    "Lfm2ComposedEndpointResidualAttentionFoldPolicy",
    "Lfm2ComposedQkAttentionAuthenticationError",
    "Lfm2ComposedQkAttentionFoldError",
    "Lfm2ComposedQkAttentionFoldPolicy",
    "compile_lfm2_composed_endpoint_residual_attention_fold",
    "compile_lfm2_composed_qk_attention_producer_fold",
    "compile_lfm2_composed_qk_stage_one_attempt",
    "compile_lfm2_composed_qk_sequence_effective_attention_fold",
]
