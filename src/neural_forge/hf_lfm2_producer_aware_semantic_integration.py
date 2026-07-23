"""Standalone semantic compiler for producer-aware LFM2 residual folding.

The compiler consumes the existing authenticated semantic trajectory package,
reuses its donor/target boundary defects and mixed hidden basis, solves one
four-sites-per-layer global closure, and absorbs virtual transports into physical
child weights.  Q/K/V, the complete short-conv B/C/X input projection, and
SwiGLU gate/up are installed as coupled atomic groups.  The final layer W2 is
never installed here; its corrected BF16 tensor is returned as a real
authenticated terminal-action prior.

Correction scale is an explicit deterministic materialization policy.  It is
candidate generation only: decoded behavioral review remains available for
every scale, and no numeric diagnostic accepts, rejects, publishes, deletes,
or stops a candidate.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from neural_forge.hf_lfm2_cpcf_terminal_action_residual import (
    Lfm2CpcfTerminalActionDownWeightPrior,
    lfm2_cpcf_terminal_action_tensor_sha256,
)
from neural_forge.hf_lfm2_global_residual_closure import (
    Lfm2GlobalResidualBasis,
    Lfm2GlobalResidualBoundaryBundle,
    Lfm2GlobalResidualClosureOperator,
    Lfm2GlobalResidualClosurePolicy,
    Lfm2GlobalResidualClosureResult,
    Lfm2GlobalResidualWriterTopology,
    fingerprint_lfm2_global_residual_tensor,
    seal_lfm2_global_residual_boundary,
    seal_lfm2_global_residual_linearization_receipt,
    solve_lfm2_global_residual_closure,
)
from neural_forge.hf_lfm2_producer_aware_global_linearization import (
    Lfm2ProducerAwareDenseTransaction,
    Lfm2ProducerAwareInstallationReceipt,
    Lfm2ProducerAwareLayerSeedWeights,
    Lfm2ProducerAwareLinearizationNormEvidence,
    Lfm2ProducerAwareWholeStackLinearizationKernel,
    build_lfm2_producer_aware_writer_topology,
    install_lfm2_producer_aware_dense_transaction,
    lfm2_producer_aware_boundary_causal_order,
    lfm2_producer_aware_whole_stack_norm_evidence,
    materialize_lfm2_producer_aware_stack_transaction,
)
from neural_forge.hf_lfm2_semantic_global_residual_integration import (
    Lfm2SemanticGlobalResidualEvidencePackage,
    build_lfm2_semantic_global_residual_basis,
    build_lfm2_semantic_global_residual_boundaries,
)


LFM2_PRODUCER_AWARE_SEMANTIC_INTEGRATION_SCHEMA = (
    "neural-forge-lfm2-producer-aware-semantic-integration-v2a"
)


class Lfm2ProducerAwareSemanticIntegrationError(RuntimeError):
    """Producer-aware semantic evidence or compilation is invalid."""


class Lfm2ProducerAwareSemanticAuthenticationError(
    Lfm2ProducerAwareSemanticIntegrationError
):
    """Producer-aware semantic evidence failed byte authentication."""


_MAX_FINITE_NORM_EVIDENCE = sys.float_info.max
_LOG_MAX_FINITE_NORM_EVIDENCE = math.log(_MAX_FINITE_NORM_EVIDENCE)


@dataclass(frozen=True, slots=True)
class _FiniteEvidenceArithmetic:
    value: float
    saturated: bool = False


def _finite_evidence_arithmetic(
    value: float | _FiniteEvidenceArithmetic,
) -> _FiniteEvidenceArithmetic:
    if isinstance(value, _FiniteEvidenceArithmetic):
        numeric = value.value
    else:
        numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise Lfm2ProducerAwareSemanticIntegrationError(
            "Producer-aware weighted evidence has a non-finite operand."
        )
    return (
        value
        if isinstance(value, _FiniteEvidenceArithmetic)
        else _FiniteEvidenceArithmetic(numeric)
    )


def _finite_saturating_evidence_product(
    left: float | _FiniteEvidenceArithmetic,
    right: float | _FiniteEvidenceArithmetic,
) -> _FiniteEvidenceArithmetic:
    left_item = _finite_evidence_arithmetic(left)
    right_item = _finite_evidence_arithmetic(right)
    left_value = left_item.value
    right_value = right_item.value
    if not all(
        math.isfinite(value) and value >= 0.0
        for value in (left_value, right_value)
    ):
        raise Lfm2ProducerAwareSemanticIntegrationError(
            "Producer-aware weighted evidence has a non-finite operand."
        )
    product = left_value * right_value
    operation_saturated = not math.isfinite(product)
    return _FiniteEvidenceArithmetic(
        _MAX_FINITE_NORM_EVIDENCE if operation_saturated else product,
        left_item.saturated or right_item.saturated or operation_saturated,
    )


def _finite_saturating_evidence_fsum(
    values: tuple[float | _FiniteEvidenceArithmetic, ...],
) -> _FiniteEvidenceArithmetic:
    checked = tuple(_finite_evidence_arithmetic(value) for value in values)
    try:
        total = math.fsum(value.value for value in checked)
    except OverflowError:
        return _FiniteEvidenceArithmetic(_MAX_FINITE_NORM_EVIDENCE, True)
    return _FiniteEvidenceArithmetic(
        total,
        any(value.saturated for value in checked),
    )


def _weighted_log_evidence(
    log_value: float | None,
    weight: float,
) -> float | None:
    if log_value is None or weight == 0.0:
        return None
    return log_value + math.log(weight)


def _log_evidence_sum(values: tuple[float | None, ...]) -> float | None:
    total = None
    for value in values:
        if value is None:
            continue
        if total is None:
            total = value
            continue
        high = max(total, value)
        low = min(total, value)
        total = high + math.log1p(math.exp(low - high))
    return total


def _finite_proxy_from_log_evidence(log_value: float | None) -> tuple[float, bool]:
    if log_value is None:
        return 0.0, False
    if log_value >= _LOG_MAX_FINITE_NORM_EVIDENCE:
        return _MAX_FINITE_NORM_EVIDENCE, True
    try:
        proxy = math.exp(log_value)
    except OverflowError:
        return _MAX_FINITE_NORM_EVIDENCE, True
    return (
        (proxy, False)
        if math.isfinite(proxy)
        else (_MAX_FINITE_NORM_EVIDENCE, True)
    )


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PyTorch is required for producer-aware semantic integration."
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
        raise Lfm2ProducerAwareSemanticIntegrationError(
            "Producer-aware semantic metadata must be canonical finite JSON."
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _seal(body: Mapping[str, Any]) -> Mapping[str, Any]:
    detached = dict(body)
    return MappingProxyType({**detached, "report_sha256": _canonical_sha256(detached)})


def verify_lfm2_producer_aware_semantic_report(
    report: Mapping[str, Any],
) -> bool:
    if not isinstance(report, Mapping):
        return False
    try:
        body = dict(report)
        supplied = body.pop("report_sha256", None)
        return isinstance(supplied, str) and supplied == _canonical_sha256(body)
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True, slots=True)
class Lfm2ProducerAwareCorrectionScalePolicy:
    """Explicit deterministic candidate frontier with no metric selection."""

    materialization_frontier: tuple[float, ...] = (1.0,)
    materialization_scale: float = 1.0
    authentication_sha256: str = ""

    def __post_init__(self) -> None:
        frontier = tuple(float(value) for value in self.materialization_frontier)
        selected = float(self.materialization_scale)
        if (
            not frontier
            or len(frontier) > 8
            or any(not math.isfinite(value) or value < 0.0 for value in frontier)
            or len(set(frontier)) != len(frontier)
            or any(
                frontier[index] >= frontier[index + 1]
                for index in range(len(frontier) - 1)
            )
            or not math.isfinite(selected)
            or selected not in frontier
        ):
            raise Lfm2ProducerAwareSemanticIntegrationError(
                "Correction scales must be a sorted unique finite nonnegative "
                "frontier of at most eight values containing materialization_scale."
            )
        object.__setattr__(self, "materialization_frontier", frontier)
        object.__setattr__(self, "materialization_scale", selected)
        expected = _canonical_sha256(self.authentication_body())
        supplied = str(self.authentication_sha256).strip().lower()
        if supplied and supplied != expected:
            raise Lfm2ProducerAwareSemanticAuthenticationError(
                "Correction-scale policy authentication is invalid."
            )
        object.__setattr__(self, "authentication_sha256", expected)

    def authentication_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_PRODUCER_AWARE_SEMANTIC_INTEGRATION_SCHEMA,
            "kind": "deterministic_correction_scale_candidate_policy",
            "materialization_frontier": list(self.materialization_frontier),
            "materialization_scale": self.materialization_scale,
            "selection_source": "explicit_policy_before_solve",
            "all_candidates_eligible_for_decoded_behavioral_review": True,
            "numeric_diagnostic_threshold": False,
            "manual_review_gate": False,
            "auto_veto": False,
        }

    def verify_authentication(self) -> bool:
        return self.authentication_sha256 == _canonical_sha256(
            self.authentication_body()
        )


def _v2_site_ids(
    evidence: Lfm2SemanticGlobalResidualEvidencePackage,
) -> tuple[str, ...]:
    ids = []
    for layer_index, macrocell in enumerate(evidence.snapshot.batches[0].macrocells):
        if macrocell.operator_kind == "full_attention":
            operator_prefix = f"model.layers.{layer_index}.self_attn"
            operator_input = f"{operator_prefix}.qkv_input_transport"
        else:
            operator_prefix = f"model.layers.{layer_index}.conv"
            operator_input = f"{operator_prefix}.bcx_input_transport"
        ids.extend(
            (
                operator_input,
                f"{operator_prefix}.out_proj.weight",
                f"model.layers.{layer_index}.feed_forward.w1_w3_input_transport",
                f"model.layers.{layer_index}.feed_forward.w2.weight",
            )
        )
    return tuple(ids)


def build_lfm2_producer_aware_semantic_topology(
    evidence: Lfm2SemanticGlobalResidualEvidencePackage,
    *,
    requested_basis_rank: int | None = None,
) -> Lfm2GlobalResidualWriterTopology:
    """Build the authenticated semantic four-sites-per-layer topology."""

    if not isinstance(evidence, Lfm2SemanticGlobalResidualEvidencePackage) or not (
        evidence.verify_authentication()
    ):
        raise Lfm2ProducerAwareSemanticAuthenticationError(
            "Cannot build producer-aware topology from mutated evidence."
        )
    return build_lfm2_producer_aware_writer_topology(
        evidence.snapshot,
        head_dimension=evidence.head_dimension,
        site_ids=_v2_site_ids(evidence),
        requested_basis_rank=requested_basis_rank,
    )


def build_lfm2_producer_aware_semantic_boundaries(
    evidence: Lfm2SemanticGlobalResidualEvidencePackage,
) -> Lfm2GlobalResidualBoundaryBundle:
    """Reuse semantic defects/masses while authenticating the v2 timeline."""

    if not isinstance(evidence, Lfm2SemanticGlobalResidualEvidencePackage) or not (
        evidence.verify_authentication()
    ):
        raise Lfm2ProducerAwareSemanticAuthenticationError(
            "Cannot build producer-aware boundaries from mutated evidence."
        )
    baseline = build_lfm2_semantic_global_residual_boundaries(evidence)
    boundaries = []
    for boundary in baseline.boundaries:
        causal_order = lfm2_producer_aware_boundary_causal_order(
            boundary.layer_index,
            boundary.boundary_kind,
        )
        boundaries.append(
            seal_lfm2_global_residual_boundary(
                boundary_id=boundary.boundary_id,
                layer_index=boundary.layer_index,
                causal_order=causal_order,
                boundary_kind=boundary.boundary_kind,
                defect=boundary.defect,
                defect_sha256=fingerprint_lfm2_global_residual_tensor(boundary.defect),
                row_mass=boundary.row_mass,
                row_mass_sha256=fingerprint_lfm2_global_residual_tensor(
                    boundary.row_mass
                ),
            )
        )
    return Lfm2GlobalResidualBoundaryBundle(tuple(boundaries))


def build_lfm2_producer_aware_semantic_basis(
    evidence: Lfm2SemanticGlobalResidualEvidencePackage,
    topology: Lfm2GlobalResidualWriterTopology,
    boundaries: Lfm2GlobalResidualBoundaryBundle,
) -> Lfm2GlobalResidualBasis:
    """Reuse the authenticated mixed defect/writer/tied-table basis builder."""

    return build_lfm2_semantic_global_residual_basis(
        evidence,
        topology,
        boundaries,
    )


def _physical_parameter_ids(
    layer_index: int,
    operator_kind: str,
) -> Mapping[str, str]:
    mlp_prefix = f"model.layers.{layer_index}.feed_forward"
    if operator_kind == "full_attention":
        prefix = f"model.layers.{layer_index}.self_attn"
        result = {
            "query_weight": f"{prefix}.q_proj.weight",
            "key_weight": f"{prefix}.k_proj.weight",
            "value_weight": f"{prefix}.v_proj.weight",
            "operator_output_weight": f"{prefix}.out_proj.weight",
        }
    else:
        prefix = f"model.layers.{layer_index}.conv"
        result = {
            "operator_input_weight": f"{prefix}.in_proj.weight",
            "operator_output_weight": f"{prefix}.out_proj.weight",
        }
    result.update(
        {
            "mlp_gate_weight": f"{mlp_prefix}.w1.weight",
            "mlp_up_weight": f"{mlp_prefix}.w3.weight",
            "mlp_down_weight": f"{mlp_prefix}.w2.weight",
        }
    )
    return MappingProxyType(result)


def _macrocell_role_weights(macrocell: Any) -> Mapping[str, Any]:
    p = macrocell.operator.parameters
    mp = macrocell.mlp.parameters
    if macrocell.operator_kind == "full_attention":
        result = {
            "query_weight": p.query_weight,
            "key_weight": p.key_weight,
            "value_weight": p.value_weight,
            "operator_output_weight": p.output_weight,
        }
    else:
        result = {
            "operator_input_weight": p.input_weight,
            "operator_output_weight": p.output_weight,
        }
    result.update(
        {
            "mlp_gate_weight": mp.gate_weight,
            "mlp_up_weight": mp.up_weight,
            "mlp_down_weight": mp.down_weight,
        }
    )
    return MappingProxyType(result)


def _validate_and_capture_physical_seeds(
    evidence: Lfm2SemanticGlobalResidualEvidencePackage,
    live_physical_parameters: Mapping[str, Any],
    deferred_final_writer_parameter: Any,
) -> tuple[tuple[Lfm2ProducerAwareLayerSeedWeights, ...], str, str]:
    """Bind live BF16 physical matrices to authenticated FP64 primals."""

    torch = _torch()
    if not evidence.verify_authentication():
        raise Lfm2ProducerAwareSemanticAuthenticationError(
            "Evidence changed before physical seed capture."
        )
    expected_live_ids = set()
    layer_ids = []
    for layer_index, macrocell in enumerate(evidence.snapshot.batches[0].macrocells):
        ids = _physical_parameter_ids(layer_index, macrocell.operator_kind)
        layer_ids.append(ids)
        expected_live_ids.update(ids.values())
    final_id = evidence.final_site_id
    expected_live_ids.remove(final_id)
    if (
        not isinstance(live_physical_parameters, Mapping)
        or set(live_physical_parameters) != expected_live_ids
    ):
        raise Lfm2ProducerAwareSemanticIntegrationError(
            "live_physical_parameters must exactly cover every nondeferred Q/K/V, "
            "conv input/output, and MLP gate/up/down matrix."
        )
    live = dict(live_physical_parameters)
    if any(
        not isinstance(value, torch.nn.Parameter)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != 2
        for value in live.values()
    ):
        raise Lfm2ProducerAwareSemanticIntegrationError(
            "Every live physical target must be an ordinary dense Parameter."
        )
    live_storage = {
        (
            value.device.type,
            value.device.index,
            int(value.untyped_storage().data_ptr()),
        )
        for value in live.values()
    }
    if len(live_storage) != len(live):
        raise Lfm2ProducerAwareSemanticIntegrationError(
            "Live producer-aware parameters cannot share tensor storage."
        )
    if (
        not isinstance(deferred_final_writer_parameter, torch.nn.Parameter)
        or deferred_final_writer_parameter.layout != torch.strided
        or deferred_final_writer_parameter.device.type == "meta"
        or deferred_final_writer_parameter.ndim != 2
    ):
        raise Lfm2ProducerAwareSemanticIntegrationError(
            "deferred_final_writer_parameter must be one dense Parameter."
        )
    final_before = lfm2_cpcf_terminal_action_tensor_sha256(
        deferred_final_writer_parameter
    )
    if final_before != evidence.deferred_final_w2_prior.live_deferred_w2_sha256:
        raise Lfm2ProducerAwareSemanticAuthenticationError(
            "Live final W2 differs from the deferred prior lineage."
        )
    final_storage = (
        deferred_final_writer_parameter.device.type,
        deferred_final_writer_parameter.device.index,
        int(deferred_final_writer_parameter.untyped_storage().data_ptr()),
    )
    if final_storage in live_storage:
        raise Lfm2ProducerAwareSemanticIntegrationError(
            "Deferred final W2 cannot share storage with an install target."
        )
    evidence_outputs = evidence.writer_seed_mapping()
    layers = []
    for layer_index, macrocell in enumerate(evidence.snapshot.batches[0].macrocells):
        ids = layer_ids[layer_index]
        expected = _macrocell_role_weights(macrocell)
        seeds = {}
        for role, parameter_id in ids.items():
            seed = (
                evidence.deferred_final_w2_prior.down_weight
                if parameter_id == final_id
                else live[parameter_id].detach()
            )
            if tuple(seed.shape) != tuple(expected[role].shape) or not bool(
                torch.equal(seed.to(device="cpu", dtype=torch.float64), expected[role])
            ):
                raise Lfm2ProducerAwareSemanticAuthenticationError(
                    f"Physical seed {parameter_id} differs from frozen primal state."
                )
            if role in {"operator_output_weight", "mlp_down_weight"}:
                authenticated_output = evidence_outputs[parameter_id]
                if fingerprint_lfm2_global_residual_tensor(seed) != (
                    fingerprint_lfm2_global_residual_tensor(authenticated_output)
                ):
                    raise Lfm2ProducerAwareSemanticAuthenticationError(
                        f"Physical output seed {parameter_id} differs from v1 evidence."
                    )
            seeds[role] = seed
        layers.append(
            Lfm2ProducerAwareLayerSeedWeights(
                layer_index=layer_index,
                operator_kind=macrocell.operator_kind,
                parameter_ids=ids,
                seed_weights=seeds,
            )
        )
    if not evidence.verify_authentication():
        raise Lfm2ProducerAwareSemanticAuthenticationError(
            "Evidence changed during physical seed capture."
        )
    return tuple(layers), final_id, final_before


def _weighted_norm_evidence(
    evidence: Lfm2SemanticGlobalResidualEvidencePackage,
    basis: Lfm2GlobalResidualBasis,
    boundaries: Lfm2GlobalResidualBoundaryBundle,
) -> tuple[float, tuple[float, ...], float, tuple[float, ...], Mapping[str, Any]]:
    conditioning, certified = lfm2_producer_aware_whole_stack_norm_evidence(
        evidence.snapshot,
        basis.output_basis,
    )
    if not isinstance(certified, Lfm2ProducerAwareLinearizationNormEvidence):
        raise Lfm2ProducerAwareSemanticIntegrationError(
            "Producer-aware norm evidence omitted its overflow contract."
        )
    maximum_coordinate_weight = max(
        float(boundary.row_mass.max().item()) for boundary in boundaries.boundaries
    ) / (len(boundaries.boundaries) * evidence.snapshot.hidden_size)
    block_scale_arithmetic = tuple(
        _finite_saturating_evidence_product(value, maximum_coordinate_weight)
        for value in conditioning.block_squared_scales
    )
    block_scales = tuple(value.value for value in block_scale_arithmetic)
    operator_scale_arithmetic = _finite_saturating_evidence_fsum(
        block_scale_arithmetic
    )
    operator_scale = operator_scale_arithmetic.value
    weighted_block_logs = tuple(
        _weighted_log_evidence(value, maximum_coordinate_weight)
        for value in certified.block_log_squared_norm_evidence
    )
    weighted_operator_log = _log_evidence_sum(weighted_block_logs)
    weighted_overflow_observed = certified.finite_float_overflow_observed
    weighted_bound_arithmetic: tuple[_FiniteEvidenceArithmetic, ...] = ()
    weighted_operator_arithmetic = _FiniteEvidenceArithmetic(0.0)
    if not weighted_overflow_observed:
        weighted_bound_arithmetic = tuple(
            _finite_saturating_evidence_product(
                value,
                maximum_coordinate_weight,
            )
            for value in certified.block_squared_norm_upper_bounds
        )
        weighted_operator_arithmetic = _finite_saturating_evidence_fsum(
            weighted_bound_arithmetic
        )
        weighted_overflow_observed = (
            weighted_operator_arithmetic.saturated
            or any(value.saturated for value in weighted_bound_arithmetic)
        )
    if weighted_overflow_observed:
        block_proxy_pairs = tuple(
            _finite_proxy_from_log_evidence(value) for value in weighted_block_logs
        )
        certified_blocks = tuple(value for value, _ in block_proxy_pairs)
        certified_operator, operator_proxy_saturated = (
            _finite_proxy_from_log_evidence(weighted_operator_log)
        )
        certified_operator = max(certified_operator, *certified_blocks)
        finite_proxy_saturated = operator_proxy_saturated or any(
            saturated for _, saturated in block_proxy_pairs
        )
        finite_bound_fields_are_certified = False
    else:
        certified_blocks = tuple(
            value.value for value in weighted_bound_arithmetic
        )
        certified_operator = weighted_operator_arithmetic.value
        finite_proxy_saturated = False
        finite_bound_fields_are_certified = True
    values = (*block_scales, operator_scale, *certified_blocks, certified_operator)
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise Lfm2ProducerAwareSemanticIntegrationError(
            "Producer-aware weighted scale evidence is non-finite."
        )
    report = _seal(
        {
            "schema": LFM2_PRODUCER_AWARE_SEMANTIC_INTEGRATION_SCHEMA,
            "kind": "producer_aware_weighted_conditioning_evidence",
            "conditioning_method": conditioning.method,
            "maximum_coordinate_weight": maximum_coordinate_weight,
            "nonuniform_row_weight_treatment": "conservative_max_W_envelope",
            "operator_squared_conditioning_scale": operator_scale,
            "block_squared_conditioning_scales": list(block_scales),
            "operator_squared_norm_upper_bound": certified_operator,
            "block_squared_norm_upper_bounds": list(certified_blocks),
            "finite_norm_bound_fields_are_certified": (
                finite_bound_fields_are_certified
            ),
            "downstream_norm_bound_float_overflow_observed": (
                weighted_overflow_observed
            ),
            "finite_norm_bound_proxy_saturated": finite_proxy_saturated,
            "operator_log_squared_norm_evidence": weighted_operator_log,
            "block_log_squared_norm_evidence": list(weighted_block_logs),
            "log_squared_norm_evidence_base": "natural",
            "overflow_norm_bound_evidence_is_advisory_only": (
                not finite_bound_fields_are_certified
            ),
            "finite_norm_bound_proxy_used_for_conditioning": False,
            "conditioning_float_overflow_observed": (
                operator_scale_arithmetic.saturated
                or any(value.saturated for value in block_scale_arithmetic)
            ),
            "conditioning_scale_separate_from_certified_bound": True,
            "numeric_scale_used_for_candidate_selection": False,
            "numeric_scale_used_for_auto_veto": False,
            "quality_gate": False,
        }
    )
    return (
        operator_scale,
        block_scales,
        certified_operator,
        certified_blocks,
        report,
    )


@dataclass(frozen=True, slots=True)
class Lfm2ProducerAwareSemanticInstallation:
    """Authenticated solved and physically installed v2a semantic closure."""

    topology: Lfm2GlobalResidualWriterTopology
    boundaries: Lfm2GlobalResidualBoundaryBundle
    basis: Lfm2GlobalResidualBasis
    closure_result: Lfm2GlobalResidualClosureResult
    dense_transaction: Lfm2ProducerAwareDenseTransaction
    installation_receipt: Lfm2ProducerAwareInstallationReceipt
    corrected_final_w2_prior: Lfm2CpcfTerminalActionDownWeightPrior
    closure_policy: Lfm2GlobalResidualClosurePolicy
    correction_scale_policy: Lfm2ProducerAwareCorrectionScalePolicy
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))
        if not self.verify_authentication():
            raise Lfm2ProducerAwareSemanticAuthenticationError(
                "Producer-aware semantic installation failed authentication."
            )

    def verify_authentication(self) -> bool:
        return (
            self.topology.verify_authentication()
            and self.boundaries.verify_authentication()
            and self.basis.verify_authentication()
            and self.closure_result.verify_authentication()
            and self.dense_transaction.verify_authentication()
            and self.installation_receipt.verify_authentication()
            and self.corrected_final_w2_prior.verify_authentication()
            and self.closure_policy.verify_authentication()
            and self.correction_scale_policy.verify_authentication()
            and verify_lfm2_producer_aware_semantic_report(self.report)
            and self.report.get("topology_sha256") == self.topology.topology_sha256
            and self.report.get("topology_receipt") == self.topology.as_dict()
            and self.report.get("requested_basis_rank")
            == self.topology.requested_basis_rank
            and self.report.get("topology_basis_rank_formula")
            == self.topology.topology_basis_rank_formula
            and self.report.get("topology_basis_rank")
            == self.topology.topology_basis_rank
            and self.report.get("effective_basis_rank")
            == self.topology.effective_basis_rank
            and self.report.get("basis_rank") == self.basis.rank
            and self.basis.rank == self.topology.effective_basis_rank
            and self.report.get("boundary_bundle_sha256")
            == self.boundaries.bundle_sha256
            and self.report.get("basis_sha256") == self.basis.basis_sha256
            and self.report.get("closure_result_sha256")
            == self.closure_result.report["report_sha256"]
            and self.report.get("dense_transaction_sha256")
            == self.dense_transaction.report["report_sha256"]
            and self.report.get("installation_receipt_sha256")
            == self.installation_receipt.report["report_sha256"]
            and self.report.get("corrected_final_w2_prior_sha256")
            == self.corrected_final_w2_prior.authentication_sha256
            and self.report.get("correction_scale_policy_sha256")
            == self.correction_scale_policy.authentication_sha256
            and self.report.get("closure_policy_sha256")
            == self.closure_policy.authentication_sha256
            and self.report.get("closure_policy") == self.closure_policy.as_dict()
            and self.report.get("materialization_compute_device")
            == self.dense_transaction.report["compute_execution_receipt"][
                "requested_device"
            ]
            and self.report.get("materialization_compute_execution_receipt_sha256")
            == self.dense_transaction.report["compute_execution_receipt_sha256"]
            and self.report.get("materialization_compute_execution_receipt")
            == self.dense_transaction.report["compute_execution_receipt"]
        )


def compile_install_lfm2_producer_aware_semantic_closure(
    evidence: Lfm2SemanticGlobalResidualEvidencePackage,
    *,
    live_physical_parameters: Mapping[str, Any],
    deferred_final_writer_parameter: Any,
    closure_policy: Lfm2GlobalResidualClosurePolicy | None = None,
    correction_scale_policy: Lfm2ProducerAwareCorrectionScalePolicy | None = None,
    materialization_compute_device: str = "cpu",
    requested_basis_rank: int | None = None,
) -> Lfm2ProducerAwareSemanticInstallation:
    """Solve v2a, install every nonfinal role, and return corrected final W2."""

    if not isinstance(evidence, Lfm2SemanticGlobalResidualEvidencePackage) or not (
        evidence.verify_authentication()
    ):
        raise Lfm2ProducerAwareSemanticAuthenticationError(
            "evidence must be an authenticated semantic residual package."
        )
    scale_policy = correction_scale_policy or (Lfm2ProducerAwareCorrectionScalePolicy())
    solve_policy = closure_policy or Lfm2GlobalResidualClosurePolicy()
    if not scale_policy.verify_authentication():
        raise Lfm2ProducerAwareSemanticAuthenticationError(
            "correction_scale_policy failed authentication."
        )
    if not solve_policy.verify_authentication():
        raise Lfm2ProducerAwareSemanticAuthenticationError(
            "closure_policy failed authentication."
        )
    layer_seeds, final_id, final_before = _validate_and_capture_physical_seeds(
        evidence,
        live_physical_parameters,
        deferred_final_writer_parameter,
    )
    topology = build_lfm2_producer_aware_semantic_topology(
        evidence,
        requested_basis_rank=requested_basis_rank,
    )
    boundaries = build_lfm2_producer_aware_semantic_boundaries(evidence)
    basis = build_lfm2_producer_aware_semantic_basis(
        evidence,
        topology,
        boundaries,
    )
    (
        operator_scale,
        block_scales,
        certified_operator,
        certified_blocks,
        scale_report,
    ) = _weighted_norm_evidence(evidence, basis, boundaries)
    linearization_receipt = seal_lfm2_global_residual_linearization_receipt(
        topology,
        boundaries,
        kernel_id="lfm2.semantic.producer-aware-whole-stack.v2a",
        source_sha256={
            "evidence_package": evidence.evidence_sha256,
            "whole_stack_snapshot": evidence.snapshot.authentication_sha256,
            "conditioning_evidence": scale_report["report_sha256"],
            "correction_scale_policy": scale_policy.authentication_sha256,
        },
        analytic_math_components=(
            "semantic_token_bridge_gather_scatter",
            "continuous_fp64_qkv_bcx_gate_up_input_transports",
            "continuous_fp64_o_convout_w2_output_writers",
            "explicit_exact_dynamic_site_whole_stack_vjp",
            "terminal_final_rmsnorm_jvp_vjp",
            "dense_absorption_without_runtime_sidecar",
        ),
        operator_squared_norm_upper_bound=certified_operator,
        block_squared_norm_upper_bounds=certified_blocks,
        operator_squared_conditioning_scale=operator_scale,
        block_squared_conditioning_scales=block_scales,
        conditioning_scale_method=("max-W-weighted-producer-aware-local-direct-energy"),
        finite_norm_bound_fields_are_certified=scale_report[
            "finite_norm_bound_fields_are_certified"
        ],
        downstream_norm_bound_float_overflow_observed=scale_report[
            "downstream_norm_bound_float_overflow_observed"
        ],
        finite_norm_bound_proxy_saturated=scale_report[
            "finite_norm_bound_proxy_saturated"
        ],
        operator_log_squared_norm_evidence=scale_report[
            "operator_log_squared_norm_evidence"
        ],
        block_log_squared_norm_evidence=scale_report[
            "block_log_squared_norm_evidence"
        ],
    )
    kernel = Lfm2ProducerAwareWholeStackLinearizationKernel(
        topology,
        boundaries,
        evidence.snapshot,
        linearization_receipt=linearization_receipt,
    )
    operator = Lfm2GlobalResidualClosureOperator(
        topology,
        basis,
        boundaries,
        kernel=kernel,
        policy=solve_policy,
    )
    closure_result = solve_lfm2_global_residual_closure(operator)
    scaled_coefficients = tuple(
        value * scale_policy.materialization_scale
        for value in closure_result.coefficient_blocks
    )
    dense_transaction = materialize_lfm2_producer_aware_stack_transaction(
        layer_seeds,
        basis.output_basis,
        scaled_coefficients,
        compute_device=materialization_compute_device,
    )
    seed_by_id = dict(
        zip(
            dense_transaction.parameter_ids,
            dense_transaction.seed_weights,
            strict=True,
        )
    )
    corrected_by_id = dict(
        zip(
            dense_transaction.parameter_ids,
            dense_transaction.corrected_weights,
            strict=True,
        )
    )
    corrected_final_weight = corrected_by_id[final_id]
    base_prior = evidence.deferred_final_w2_prior
    corrected_final_prior = Lfm2CpcfTerminalActionDownWeightPrior(
        down_weight=corrected_final_weight,
        body_recipe_id=base_prior.body_recipe_id,
        body_recipe_sha256=base_prior.body_recipe_sha256,
        mlp_lane=base_prior.mlp_lane,
        selected_role=base_prior.selected_role,
        selected_candidate_sha256=base_prior.selected_candidate_sha256,
        preparation_sha256=base_prior.preparation_sha256,
        mlp_installation_core_sha256=base_prior.mlp_installation_core_sha256,
        live_deferred_w2_sha256=base_prior.live_deferred_w2_sha256,
    )
    if not evidence.verify_authentication():
        raise Lfm2ProducerAwareSemanticAuthenticationError(
            "Evidence changed before producer-aware dense publication."
        )
    installation_receipt = install_lfm2_producer_aware_dense_transaction(
        dense_transaction,
        live_physical_parameters,
        deferred_parameter_ids=(final_id,),
    )
    final_after = lfm2_cpcf_terminal_action_tensor_sha256(
        deferred_final_writer_parameter
    )
    if final_after != final_before:
        torch = _torch()
        with torch.no_grad():
            for parameter_id, parameter in live_physical_parameters.items():
                parameter.copy_(
                    seed_by_id[parameter_id].to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                )
        raise Lfm2ProducerAwareSemanticAuthenticationError(
            "Deferred final W2 changed; every installed producer-aware role was restored."
        )
    installed_parameter_ids = tuple(installation_receipt.post_install_sha256)
    dense_live_before = {
        parameter_id: fingerprint_lfm2_global_residual_tensor(seed_by_id[parameter_id])
        for parameter_id in installed_parameter_ids
    }
    dense_live_after = dict(installation_receipt.post_install_sha256)
    changed_parameter_ids = tuple(
        sorted(
            parameter_id
            for parameter_id in installed_parameter_ids
            if dense_live_before[parameter_id] != dense_live_after[parameter_id]
        )
    )
    physical_parameter_specifications = [
        {
            "parameter_id": parameter_id,
            "layer_index": layer_index,
            "role": role,
            "seed_sha256": fingerprint_lfm2_global_residual_tensor(seed),
            "corrected_sha256": fingerprint_lfm2_global_residual_tensor(corrected),
            "installed": parameter_id != final_id,
            "deferred": parameter_id == final_id,
        }
        for parameter_id, layer_index, role, seed, corrected in zip(
            dense_transaction.parameter_ids,
            dense_transaction.layer_indices,
            dense_transaction.roles,
            dense_transaction.seed_weights,
            dense_transaction.corrected_weights,
            strict=True,
        )
    ]
    final_prior_lineage = _seal(
        {
            "schema": LFM2_PRODUCER_AWARE_SEMANTIC_INTEGRATION_SCHEMA,
            "kind": "producer_aware_closure_corrected_terminal_action_prior_lineage",
            "base_qdeim_prior_authentication_sha256": (
                base_prior.authentication_sha256
            ),
            "corrected_terminal_action_prior_authentication_sha256": (
                corrected_final_prior.authentication_sha256
            ),
            "closure_result_sha256": closure_result.report["report_sha256"],
            "dense_transaction_sha256": dense_transaction.report["report_sha256"],
            "correction_scale_policy_sha256": scale_policy.authentication_sha256,
            "topology_sha256": topology.topology_sha256,
            "site_id": final_id,
            "down_weight_sha256": lfm2_cpcf_terminal_action_tensor_sha256(
                corrected_final_prior.down_weight
            ),
            "dtype": str(corrected_final_weight.dtype).removeprefix("torch."),
            "shape": [int(value) for value in corrected_final_weight.shape],
            "installed": False,
            "installation_count": 0,
            "live_final_w2_sha256": final_before,
            "decoded_behavior_review_is_evidence_for_understanding": True,
            "verdict_required_for_continuation": False,
            "quality_gate": False,
        }
    )
    body = {
        "schema": LFM2_PRODUCER_AWARE_SEMANTIC_INTEGRATION_SCHEMA,
        "kind": "producer_aware_semantic_global_residual_installation",
        "evidence_package_sha256": evidence.evidence_sha256,
        "topology_sha256": topology.topology_sha256,
        "requested_basis_rank": topology.requested_basis_rank,
        "topology_basis_rank_formula": topology.topology_basis_rank_formula,
        "topology_basis_rank": topology.topology_basis_rank,
        "effective_basis_rank": topology.effective_basis_rank,
        "basis_rank": basis.rank,
        "boundary_bundle_sha256": boundaries.bundle_sha256,
        "basis_sha256": basis.basis_sha256,
        "linearization_receipt_sha256": linearization_receipt["report_sha256"],
        "kernel_authentication_sha256": kernel.authentication_sha256,
        "closure_result_sha256": closure_result.report["report_sha256"],
        "dense_transaction_sha256": dense_transaction.report["report_sha256"],
        "materialization_compute_device": materialization_compute_device,
        "materialization_compute_execution_receipt_sha256": (
            dense_transaction.report["compute_execution_receipt_sha256"]
        ),
        "materialization_compute_execution_receipt": dict(
            dense_transaction.report["compute_execution_receipt"]
        ),
        "installation_receipt_sha256": installation_receipt.report["report_sha256"],
        "corrected_final_w2_prior_sha256": (
            corrected_final_prior.authentication_sha256
        ),
        "correction_scale_policy_sha256": scale_policy.authentication_sha256,
        "correction_scale_policy": {
            **scale_policy.authentication_body(),
            "authentication_sha256": scale_policy.authentication_sha256,
        },
        "conditioning_evidence_sha256": scale_report["report_sha256"],
        "closure_policy_sha256": solve_policy.authentication_sha256,
        "closure_policy": solve_policy.as_dict(),
        "trajectory_count": len(evidence.trajectories),
        "layer_count": evidence.snapshot.layer_count,
        "dynamic_layer_count_derived_from_authenticated_evidence": True,
        "writer_site_count": len(topology.sites),
        "four_sites_per_layer": len(topology.sites)
        == 4 * evidence.snapshot.layer_count,
        "input_transport_site_count": 2 * evidence.snapshot.layer_count,
        "output_writer_site_count": 2 * evidence.snapshot.layer_count,
        "site_specifications": [site.as_dict() for site in topology.sites],
        "topology_receipt": topology.as_dict(),
        "dense_transaction_receipt": dict(dense_transaction.report),
        "physical_installation_receipt": dict(installation_receipt.report),
        "physical_parameter_specifications": physical_parameter_specifications,
        "closure_backend": "producer_aware_v2a",
        "global_residual_backend": solve_policy.backend,
        "closure_scope": (
            f"{4 * evidence.snapshot.layer_count}_producer_aware_input_output_sites_v2a"
        ),
        "attention_parameterization": "coupled_QKV_right_absorption_Wrole@U@C",
        "short_conv_parameterization": "coupled_all_BCX_rows_Win@U@C",
        "mlp_parameterization": "coupled_gate_up_right_absorption_Wrole@U@C",
        "output_parameterization": "U@C",
        "materialization_scale": scale_policy.materialization_scale,
        "materialization_frontier": list(scale_policy.materialization_frontier),
        "materialization_scale_is_execution_variant_not_winner": True,
        "frontier_member_selected_as_winner": False,
        "separate_fold_decode_retain_required_for_each_frontier_scale": True,
        "all_frontier_candidates_eligible_for_decoded_behavioral_review": True,
        "selected_by_numeric_metric": False,
        "selected_by_manual_gate": False,
        "numeric_or_manual_quality_gate": False,
        "numeric_diagnostics_used_for_auto_veto": False,
        "decoded_behavior_review_is_evidence_for_understanding": True,
        "verdict_required_for_continuation": False,
        "accept_veto_publish_delete_stop_authority": {
            "decoded_behavior_review": False,
            "numeric_metrics": False,
        },
        "installed_parameter_ids": list(installed_parameter_ids),
        "installed_parameter_count": len(installed_parameter_ids),
        "dense_live_physical_sha256_before": dense_live_before,
        "dense_live_physical_sha256_after": dense_live_after,
        "changed_physical_parameter_ids": list(changed_parameter_ids),
        "all_nondeferred_physical_parameters_installed": True,
        "final_w2_parameter_id": final_id,
        "final_w2_live_sha256_before": final_before,
        "final_w2_live_sha256_after": final_after,
        "final_w2_unchanged": final_after == final_before,
        "final_w2_installation_count": 0,
        "final_w2_corrected_prior_returned": True,
        "corrected_final_w2_closure_lineage_sha256": final_prior_lineage[
            "report_sha256"
        ],
        "final_layer_mlp_down_weight_prior_authentication_sha256": (
            corrected_final_prior.authentication_sha256
        ),
        "final_layer_mlp_down_weight_prior_tensor_sha256": (
            lfm2_cpcf_terminal_action_tensor_sha256(corrected_final_prior.down_weight)
        ),
        "final_layer_mlp_down_weight_prior_live_w2_sha256": (
            corrected_final_prior.live_deferred_w2_sha256
        ),
        "final_layer_mlp_down_weight_prior_installed_before_terminal": False,
        "final_layer_mlp_down_weight_prior_install_count": 0,
        "final_layer_mlp_down_weight_prior_closure_lineage_sha256": (
            final_prior_lineage["report_sha256"]
        ),
        "runtime_sidecar_required": False,
        "autograd_grad_calls": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "training_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
        "quality_gate": False,
    }
    return Lfm2ProducerAwareSemanticInstallation(
        topology=topology,
        boundaries=boundaries,
        basis=basis,
        closure_result=closure_result,
        dense_transaction=dense_transaction,
        installation_receipt=installation_receipt,
        corrected_final_w2_prior=corrected_final_prior,
        closure_policy=solve_policy,
        correction_scale_policy=scale_policy,
        report=_seal(body),
    )


def integrate_lfm2_producer_aware_global_closure(
    evidence: Lfm2SemanticGlobalResidualEvidencePackage,
    *,
    live_physical_parameters: Mapping[str, Any],
    deferred_final_writer_parameter: Any,
    closure_policy: Lfm2GlobalResidualClosurePolicy | None = None,
    correction_scale_policy: Lfm2ProducerAwareCorrectionScalePolicy | None = None,
    materialization_compute_device: str = "cpu",
    requested_basis_rank: int | None = None,
) -> Lfm2ProducerAwareSemanticInstallation:
    """Canonical named entrypoint for producer-aware physical residual closure."""

    return compile_install_lfm2_producer_aware_semantic_closure(
        evidence,
        live_physical_parameters=live_physical_parameters,
        deferred_final_writer_parameter=deferred_final_writer_parameter,
        closure_policy=closure_policy,
        correction_scale_policy=correction_scale_policy,
        materialization_compute_device=materialization_compute_device,
        requested_basis_rank=requested_basis_rank,
    )


__all__ = [
    "LFM2_PRODUCER_AWARE_SEMANTIC_INTEGRATION_SCHEMA",
    "Lfm2ProducerAwareCorrectionScalePolicy",
    "Lfm2ProducerAwareSemanticAuthenticationError",
    "Lfm2ProducerAwareSemanticInstallation",
    "Lfm2ProducerAwareSemanticIntegrationError",
    "build_lfm2_producer_aware_semantic_basis",
    "build_lfm2_producer_aware_semantic_boundaries",
    "build_lfm2_producer_aware_semantic_topology",
    "compile_install_lfm2_producer_aware_semantic_closure",
    "integrate_lfm2_producer_aware_global_closure",
    "verify_lfm2_producer_aware_semantic_report",
]
