"""Authenticated prepared-state APIs for repeated cached second-order options.

Preparation computes source-independent evidence statistics and each requested
parent/child mismatch target once.  Callers can then encode one grid point at a
time and immediately discard its dense reconstruction.  The emitted option is
byte- and objective-identical to the corresponding point from the original
multi-point solver.

For joint V/O attention, the value-side and baseline output-side statistics
are invariant and prepared once.  The output correction after a quantized V
candidate remains mathematically candidate-dependent; those branch-local
statistics and targets are intentionally recomputed and explicitly audited by
the streaming consumer rather than being mislabeled as invariant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from torch import Tensor

from .hf_lfm2_cached_second_order_attention_solver import (
    Lfm2CachedSecondOrderError,
    Lfm2CachedSecondOrderEvidence,
    Lfm2JointAttentionConfig,
    Lfm2JointAttentionOption,
    Lfm2SealedAttentionBridge,
    Lfm2SecondOrderOpsAudit,
    Lfm2SecondOrderQuantizedOption,
    Lfm2SecondOrderStatistics,
    _attention_context_from_values,
    _candidate_objective,
    _configuration_sha256,
    _corrected_target,
    _grid_configs,
    _joint_branch,
    _ppm,
    _quantize_solved_tensor,
    _sha256_canonical,
    _source_weight,
    _tensor_fingerprint,
    _validated_axis,
    _verify_statistics,
    compute_lfm2_second_order_statistics,
    verify_lfm2_attention_bridge,
    verify_lfm2_cached_second_order_evidence,
)


_PREPARED_SITE_SCHEMA = "neural-forge.lfm2.prepared-second-order-site.v1"
_PREPARED_JOINT_SCHEMA = "neural-forge.lfm2.prepared-joint-attention.v1"


@dataclass(frozen=True, slots=True)
class Lfm2PreparedComputationAudit:
    """Exact invariant work performed before grid-option enumeration."""

    invariant_statistics_build_count: int
    invariant_target_build_count: int
    invariant_input_factorization_build_count: int
    invariant_output_factorization_build_count: int
    candidate_dependent_statistics_per_option: int
    candidate_dependent_targets_per_option: int
    parent_model_calls: int = 0
    child_model_calls: int = 0
    forward_calls: int = 0
    generation_calls: int = 0
    gradient_calls: int = 0
    optimizer_steps: int = 0
    training_steps: int = 0


@dataclass(frozen=True, slots=True)
class Lfm2PreparedMismatchTarget:
    """One authenticated full-precision target for a mismatch coefficient."""

    mismatch_coefficient_ppm: int
    target_weight: Tensor = field(repr=False, compare=False)
    full_precision_objective: float
    target_sha256: str


@dataclass(frozen=True, slots=True)
class Lfm2PreparedInverseCholeskyBlock:
    """One authenticated inverse-Cholesky output-row block."""

    start_row: int
    stop_row: int
    inverse_cholesky: Tensor = field(repr=False, compare=False)
    factor_sha256: str


@dataclass(frozen=True, slots=True)
class Lfm2PreparedSecondOrderSite:
    """Reusable ordinary-site statistics, source, and mismatch targets."""

    schema: str
    name: str
    role: str
    source_weight: Tensor = field(repr=False, compare=False)
    source_sha256: str
    prepared_source_sha256: str
    evidence: Lfm2CachedSecondOrderEvidence = field(repr=False, compare=False)
    statistics: Lfm2SecondOrderStatistics = field(repr=False, compare=False)
    input_inverse_cholesky: Tensor = field(repr=False, compare=False)
    input_inverse_cholesky_sha256: str
    output_inverse_cholesky_blocks: tuple[
        Lfm2PreparedInverseCholeskyBlock,
        ...,
    ] = field(repr=False, compare=False)
    targets: tuple[Lfm2PreparedMismatchTarget, ...] = field(
        repr=False,
        compare=False,
    )
    current_objective: float
    row_block_size: int
    prepared_sha256: str
    computation_audit: Lfm2PreparedComputationAudit


@dataclass(frozen=True, slots=True)
class Lfm2PreparedJointAttention:
    """Reusable invariant state for one fixed-A attention V/O pair."""

    schema: str
    value_name: str
    output_name: str
    value_role: str
    output_role: str
    value_source: Tensor = field(repr=False, compare=False)
    output_source: Tensor = field(repr=False, compare=False)
    value_source_sha256: str
    output_source_sha256: str
    prepared_value_source_sha256: str
    prepared_output_source_sha256: str
    value_evidence: Lfm2CachedSecondOrderEvidence = field(
        repr=False,
        compare=False,
    )
    output_evidence: Lfm2CachedSecondOrderEvidence = field(
        repr=False,
        compare=False,
    )
    attention_bridge: Lfm2SealedAttentionBridge = field(
        repr=False,
        compare=False,
    )
    value_statistics: Lfm2SecondOrderStatistics = field(
        repr=False,
        compare=False,
    )
    output_baseline_statistics: Lfm2SecondOrderStatistics = field(
        repr=False,
        compare=False,
    )
    value_input_inverse_cholesky: Tensor = field(
        repr=False,
        compare=False,
    )
    value_input_inverse_cholesky_sha256: str
    value_output_inverse_cholesky_blocks: tuple[
        Lfm2PreparedInverseCholeskyBlock,
        ...,
    ] = field(repr=False, compare=False)
    value_targets: tuple[Lfm2PreparedMismatchTarget, ...] = field(
        repr=False,
        compare=False,
    )
    current_joint_objective: float
    value_objective_weight_ppm: int
    damping_ratio: float
    parent_input_blend: float
    output_damping_ratio: float
    row_block_size: int
    evidence_bundle_sha256: str
    source_bundle_sha256: str
    prepared_sha256: str
    computation_audit: Lfm2PreparedComputationAudit


def _inverse_cholesky(value: Tensor) -> Tensor:
    inverse = torch.cholesky_inverse(torch.linalg.cholesky(value))
    return torch.linalg.cholesky(inverse, upper=True)


def _output_inverse_blocks(
    h_out: Tensor | None,
    *,
    rows: int,
    row_block_size: int,
) -> tuple[Lfm2PreparedInverseCholeskyBlock, ...]:
    if h_out is None:
        return ()
    output = []
    for start in range(0, rows, row_block_size):
        stop = min(start + row_block_size, rows)
        factor = _inverse_cholesky(h_out[start:stop, start:stop])
        output.append(
            Lfm2PreparedInverseCholeskyBlock(
                start_row=start,
                stop_row=stop,
                inverse_cholesky=factor,
                factor_sha256=_tensor_fingerprint(factor),
            )
        )
    return tuple(output)


def _block_body(
    block: Lfm2PreparedInverseCholeskyBlock,
) -> dict[str, Any]:
    return {
        "start_row": block.start_row,
        "stop_row": block.stop_row,
        "factor_sha256": block.factor_sha256,
    }


def _solver_blocks(
    blocks: Sequence[Lfm2PreparedInverseCholeskyBlock],
) -> tuple[tuple[int, int, Tensor], ...]:
    return tuple(
        (block.start_row, block.stop_row, block.inverse_cholesky)
        for block in blocks
    )


def _verify_inverse_blocks(
    blocks: Sequence[Lfm2PreparedInverseCholeskyBlock],
    *,
    rows: int,
    row_block_size: int,
    expected: bool,
) -> None:
    expected_ranges = (
        tuple(
            (start, min(start + row_block_size, rows))
            for start in range(0, rows, row_block_size)
        )
        if expected
        else ()
    )
    observed_ranges = tuple(
        (block.start_row, block.stop_row) for block in blocks
    )
    if observed_ranges != expected_ranges:
        raise Lfm2CachedSecondOrderError(
            "Prepared output inverse-Cholesky block geometry changed."
        )
    for block in blocks:
        size = block.stop_row - block.start_row
        factor = block.inverse_cholesky
        if (
            factor.shape != (size, size)
            or factor.dtype != torch.float64
            or factor.device.type != "cpu"
            or not bool(torch.isfinite(factor).all())
            or bool(torch.diag(factor).le(0).any())
            or _tensor_fingerprint(factor) != block.factor_sha256
        ):
            raise Lfm2CachedSecondOrderError(
                "Prepared output inverse-Cholesky authentication failed."
            )


def _target_map(
    targets: Sequence[Lfm2PreparedMismatchTarget],
) -> dict[int, Lfm2PreparedMismatchTarget]:
    output = {target.mismatch_coefficient_ppm: target for target in targets}
    if len(output) != len(targets) or tuple(output) != tuple(sorted(output)):
        raise Lfm2CachedSecondOrderError(
            "Prepared mismatch targets must be unique and sorted."
        )
    return output


def _prepared_target_body(
    target: Lfm2PreparedMismatchTarget,
) -> dict[str, Any]:
    return {
        "mismatch_coefficient_ppm": target.mismatch_coefficient_ppm,
        "full_precision_objective": target.full_precision_objective,
        "target_sha256": target.target_sha256,
    }


def _prepared_site_body(
    prepared: Lfm2PreparedSecondOrderSite,
    targets: Sequence[Lfm2PreparedMismatchTarget],
) -> dict[str, Any]:
    return {
        "schema": prepared.schema,
        "name": prepared.name,
        "role": prepared.role,
        "source_sha256": prepared.source_sha256,
        "prepared_source_sha256": prepared.prepared_source_sha256,
        "evidence_sha256": prepared.evidence.evidence_sha256,
        "statistics_sha256": prepared.statistics.statistics_sha256,
        "input_inverse_cholesky_sha256": (
            prepared.input_inverse_cholesky_sha256
        ),
        "output_inverse_cholesky_blocks": tuple(
            _block_body(value)
            for value in prepared.output_inverse_cholesky_blocks
        ),
        "targets": tuple(_prepared_target_body(value) for value in targets),
        "current_objective": prepared.current_objective,
        "row_block_size": prepared.row_block_size,
    }


def _prepared_joint_body(
    prepared: Lfm2PreparedJointAttention,
    targets: Sequence[Lfm2PreparedMismatchTarget],
) -> dict[str, Any]:
    return {
        "schema": prepared.schema,
        "value_name": prepared.value_name,
        "output_name": prepared.output_name,
        "value_role": prepared.value_role,
        "output_role": prepared.output_role,
        "value_source_sha256": prepared.value_source_sha256,
        "output_source_sha256": prepared.output_source_sha256,
        "prepared_value_source_sha256": (prepared.prepared_value_source_sha256),
        "prepared_output_source_sha256": (prepared.prepared_output_source_sha256),
        "value_evidence_sha256": prepared.value_evidence.evidence_sha256,
        "output_evidence_sha256": prepared.output_evidence.evidence_sha256,
        "attention_bridge_sha256": prepared.attention_bridge.bridge_sha256,
        "value_statistics_sha256": (prepared.value_statistics.statistics_sha256),
        "output_baseline_statistics_sha256": (
            prepared.output_baseline_statistics.statistics_sha256
        ),
        "value_input_inverse_cholesky_sha256": (
            prepared.value_input_inverse_cholesky_sha256
        ),
        "value_output_inverse_cholesky_blocks": tuple(
            _block_body(value)
            for value in prepared.value_output_inverse_cholesky_blocks
        ),
        "value_targets": tuple(_prepared_target_body(value) for value in targets),
        "current_joint_objective": prepared.current_joint_objective,
        "value_objective_weight_ppm": prepared.value_objective_weight_ppm,
        "damping_ratio": prepared.damping_ratio,
        "parent_input_blend": prepared.parent_input_blend,
        "output_damping_ratio": prepared.output_damping_ratio,
        "row_block_size": prepared.row_block_size,
        "evidence_bundle_sha256": prepared.evidence_bundle_sha256,
        "source_bundle_sha256": prepared.source_bundle_sha256,
    }


def _mismatch_ppm_values(
    mismatch_coefficients: Sequence[float],
) -> tuple[int, ...]:
    return tuple(
        sorted(
            _ppm(
                value,
                "mismatch_coefficient",
                minimum=-2.0,
                maximum=2.0,
            )
            for value in _validated_axis(
                mismatch_coefficients,
                "mismatch_coefficients",
            )
        )
    )


def prepare_lfm2_second_order_site(
    weight: Tensor,
    evidence: Lfm2CachedSecondOrderEvidence,
    *,
    name: str,
    role: str,
    mismatch_coefficients: Sequence[float] = (0.0, 1.0),
    damping_ratio: float = 1e-4,
    parent_input_blend: float = 0.5,
    output_damping_ratio: float = 1e-4,
    row_block_size: int = 32,
) -> Lfm2PreparedSecondOrderSite:
    """Authenticate and compute all invariant ordinary-site work once."""

    verify_lfm2_cached_second_order_evidence(evidence)
    source, source_sha = _source_weight(weight, "weight")
    if source.shape != (evidence.output_features, evidence.input_features):
        raise Lfm2CachedSecondOrderError(
            "Weight geometry does not match the sealed evidence."
        )
    if not isinstance(name, str) or not name.strip():
        raise Lfm2CachedSecondOrderError("name must be a nonempty string.")
    if not isinstance(role, str) or not role.strip():
        raise Lfm2CachedSecondOrderError("role must be a nonempty string.")
    mismatch_ppm = _mismatch_ppm_values(mismatch_coefficients)
    statistics = compute_lfm2_second_order_statistics(
        evidence,
        damping_ratio=damping_ratio,
        parent_input_blend=parent_input_blend,
        output_damping_ratio=output_damping_ratio,
    )
    input_inverse_cholesky = _inverse_cholesky(statistics.h_in)
    input_inverse_cholesky_sha = _tensor_fingerprint(
        input_inverse_cholesky
    )
    output_inverse_blocks = _output_inverse_blocks(
        statistics.h_out,
        rows=int(source.shape[0]),
        row_block_size=row_block_size,
    )
    current = _candidate_objective(
        source,
        source,
        evidence,
        statistics.h_out,
    )
    targets = []
    for coefficient_ppm in mismatch_ppm:
        target = _corrected_target(
            source,
            evidence,
            statistics,
            mismatch_coefficient=coefficient_ppm / 1_000_000,
            row_block_size=row_block_size,
        )
        targets.append(
            Lfm2PreparedMismatchTarget(
                mismatch_coefficient_ppm=coefficient_ppm,
                target_weight=target,
                full_precision_objective=_candidate_objective(
                    source,
                    target,
                    evidence,
                    statistics.h_out,
                ),
                target_sha256=_tensor_fingerprint(target),
            )
        )
    prepared_source_sha = _tensor_fingerprint(source)
    audit = Lfm2PreparedComputationAudit(
        invariant_statistics_build_count=1,
        invariant_target_build_count=len(targets),
        invariant_input_factorization_build_count=1,
        invariant_output_factorization_build_count=len(
            output_inverse_blocks
        ),
        candidate_dependent_statistics_per_option=0,
        candidate_dependent_targets_per_option=0,
    )
    provisional = Lfm2PreparedSecondOrderSite(
        schema=_PREPARED_SITE_SCHEMA,
        name=name.strip(),
        role=role.strip(),
        source_weight=source,
        source_sha256=source_sha,
        prepared_source_sha256=prepared_source_sha,
        evidence=evidence,
        statistics=statistics,
        input_inverse_cholesky=input_inverse_cholesky,
        input_inverse_cholesky_sha256=input_inverse_cholesky_sha,
        output_inverse_cholesky_blocks=output_inverse_blocks,
        targets=tuple(targets),
        current_objective=current,
        row_block_size=row_block_size,
        prepared_sha256="",
        computation_audit=audit,
    )
    return Lfm2PreparedSecondOrderSite(
        schema=provisional.schema,
        name=provisional.name,
        role=provisional.role,
        source_weight=source,
        source_sha256=source_sha,
        prepared_source_sha256=prepared_source_sha,
        evidence=evidence,
        statistics=statistics,
        input_inverse_cholesky=input_inverse_cholesky,
        input_inverse_cholesky_sha256=input_inverse_cholesky_sha,
        output_inverse_cholesky_blocks=output_inverse_blocks,
        targets=tuple(targets),
        current_objective=current,
        row_block_size=row_block_size,
        prepared_sha256=_sha256_canonical(_prepared_site_body(provisional, targets)),
        computation_audit=audit,
    )


def verify_lfm2_prepared_second_order_site(
    prepared: Lfm2PreparedSecondOrderSite,
) -> None:
    """Fail closed if prepared ordinary-site state changed."""

    if not isinstance(prepared, Lfm2PreparedSecondOrderSite):
        raise Lfm2CachedSecondOrderError(
            "prepared must be Lfm2PreparedSecondOrderSite."
        )
    if prepared.schema != _PREPARED_SITE_SCHEMA:
        raise Lfm2CachedSecondOrderError("Unsupported prepared-site schema.")
    verify_lfm2_cached_second_order_evidence(prepared.evidence)
    _verify_statistics(prepared.statistics, prepared.evidence)
    if (
        prepared.source_weight.dtype != torch.float64
        or prepared.source_weight.device.type != "cpu"
        or prepared.source_weight.shape
        != (
            prepared.evidence.output_features,
            prepared.evidence.input_features,
        )
        or _tensor_fingerprint(prepared.source_weight)
        != prepared.prepared_source_sha256
    ):
        raise Lfm2CachedSecondOrderError(
            "Prepared source content authentication failed."
        )
    if (
        prepared.input_inverse_cholesky.shape
        != prepared.statistics.h_in.shape
        or prepared.input_inverse_cholesky.dtype != torch.float64
        or prepared.input_inverse_cholesky.device.type != "cpu"
        or not bool(
            torch.isfinite(prepared.input_inverse_cholesky).all()
        )
        or bool(torch.diag(prepared.input_inverse_cholesky).le(0).any())
        or _tensor_fingerprint(prepared.input_inverse_cholesky)
        != prepared.input_inverse_cholesky_sha256
    ):
        raise Lfm2CachedSecondOrderError(
            "Prepared input inverse-Cholesky authentication failed."
        )
    _verify_inverse_blocks(
        prepared.output_inverse_cholesky_blocks,
        rows=int(prepared.source_weight.shape[0]),
        row_block_size=prepared.row_block_size,
        expected=prepared.statistics.h_out is not None,
    )
    targets = _target_map(prepared.targets)
    for coefficient_ppm, target in targets.items():
        if (
            target.target_weight.shape != prepared.source_weight.shape
            or target.target_weight.dtype != torch.float64
            or target.target_weight.device.type != "cpu"
            or _tensor_fingerprint(target.target_weight) != target.target_sha256
            or coefficient_ppm != target.mismatch_coefficient_ppm
        ):
            raise Lfm2CachedSecondOrderError(
                "Prepared mismatch target authentication failed."
            )
    seal = _sha256_canonical(_prepared_site_body(prepared, prepared.targets))
    if seal != prepared.prepared_sha256:
        raise Lfm2CachedSecondOrderError(
            "Prepared-site metadata authentication failed."
        )


def solve_lfm2_prepared_second_order_option(
    prepared: Lfm2PreparedSecondOrderSite,
    *,
    bit_width: int,
    group_size: int,
    clipping_ratio: float,
    rounding_offset: float,
    outliers_per_group: int,
    mismatch_coefficient: float,
) -> Lfm2SecondOrderQuantizedOption:
    """Encode one ordinary grid point from authenticated prepared state."""

    verify_lfm2_prepared_second_order_site(prepared)
    configs = _grid_configs(
        input_features=prepared.evidence.input_features,
        bit_widths=(bit_width,),
        group_sizes=(group_size,),
        clipping_ratios=(clipping_ratio,),
        rounding_offsets=(rounding_offset,),
        outliers_per_group=(outliers_per_group,),
        mismatch_coefficients=(mismatch_coefficient,),
    )
    if len(configs) != 1:
        raise Lfm2CachedSecondOrderError(
            "A prepared option requires exactly one grid point."
        )
    config = configs[0]
    try:
        target = _target_map(prepared.targets)[config.mismatch_coefficient_ppm]
    except KeyError as exc:
        raise Lfm2CachedSecondOrderError(
            "The requested mismatch target was not prepared."
        ) from exc
    statistics = prepared.statistics
    source = prepared.source_weight
    rtn_payload = _quantize_solved_tensor(
        source_weight=source,
        target_weight=target.target_weight,
        h_in=statistics.h_in,
        h_out=statistics.h_out,
        output_row_block_size=prepared.row_block_size,
        name=prepared.name,
        role=prepared.role,
        config=config,
        successive=False,
        source_sha256=prepared.source_sha256,
        prepared_input_inverse_cholesky=(
            prepared.input_inverse_cholesky
        ),
        prepared_output_inverse_cholesky_blocks=_solver_blocks(
            prepared.output_inverse_cholesky_blocks
        ),
    )
    rtn_objective = _candidate_objective(
        source,
        rtn_payload.reconstructed_tensor,
        prepared.evidence,
        statistics.h_out,
    )
    successive_payload = _quantize_solved_tensor(
        source_weight=source,
        target_weight=target.target_weight,
        h_in=statistics.h_in,
        h_out=statistics.h_out,
        output_row_block_size=prepared.row_block_size,
        name=prepared.name,
        role=prepared.role,
        config=config,
        successive=True,
        source_sha256=prepared.source_sha256,
        prepared_input_inverse_cholesky=(
            prepared.input_inverse_cholesky
        ),
        prepared_output_inverse_cholesky_blocks=_solver_blocks(
            prepared.output_inverse_cholesky_blocks
        ),
    )
    successive_objective = _candidate_objective(
        source,
        successive_payload.reconstructed_tensor,
        prepared.evidence,
        statistics.h_out,
    )
    tolerance = 1e-12 * max(1.0, abs(rtn_objective))
    if successive_objective <= rtn_objective + tolerance:
        selected_method = "successive"
        selected_payload = successive_payload
        selected_objective = successive_objective
    else:
        selected_method = "rtn_rollback"
        selected_payload = rtn_payload
        selected_objective = rtn_objective
    config_sha = _configuration_sha256(config)
    option_id = f"{prepared.name}:{config.option_suffix}"
    option_sha = _sha256_canonical(
        {
            "option_id": option_id,
            "config_sha256": config_sha,
            "evidence_sha256": prepared.evidence.evidence_sha256,
            "statistics_sha256": statistics.statistics_sha256,
            "source_sha256": prepared.source_sha256,
            "physical_payload_sha256": (selected_payload.physical_payload_sha256),
            "selected_method": selected_method,
            "selected_objective": selected_objective,
            "rtn_physical_payload_sha256": (rtn_payload.physical_payload_sha256),
            "raw_successive_physical_payload_sha256": (
                successive_payload.physical_payload_sha256
            ),
        }
    )
    audit = Lfm2SecondOrderOpsAudit(
        output_hessian_row_compensation=statistics.h_out is not None,
    )
    result = Lfm2SecondOrderQuantizedOption(
        option_id=option_id,
        config=config,
        payload=selected_payload,
        selected_method=selected_method,
        current_objective=prepared.current_objective,
        full_precision_objective=target.full_precision_objective,
        rtn_objective=rtn_objective,
        raw_successive_objective=successive_objective,
        selected_objective=selected_objective,
        improvement_vs_current=(prepared.current_objective - selected_objective),
        improvement_vs_rtn=rtn_objective - selected_objective,
        evidence_sha256=prepared.evidence.evidence_sha256,
        statistics_sha256=statistics.statistics_sha256,
        source_sha256=prepared.source_sha256,
        config_sha256=config_sha,
        rtn_physical_payload_sha256=(rtn_payload.physical_payload_sha256),
        raw_successive_physical_payload_sha256=(
            successive_payload.physical_payload_sha256
        ),
        option_sha256=option_sha,
        ops_audit=audit,
    )
    verify_lfm2_prepared_second_order_site(prepared)
    return result


def prepare_lfm2_joint_attention_vo(
    value_weight: Tensor,
    output_weight: Tensor,
    value_evidence: Lfm2CachedSecondOrderEvidence,
    output_evidence: Lfm2CachedSecondOrderEvidence,
    *,
    attention_bridge: Lfm2SealedAttentionBridge,
    value_name: str,
    output_name: str,
    value_role: str = "attention.v",
    output_role: str = "attention.o",
    mismatch_coefficients: Sequence[float] = (0.0, 1.0),
    value_objective_weight: float = 0.1,
    damping_ratio: float = 1e-4,
    parent_input_blend: float = 0.5,
    output_damping_ratio: float = 1e-4,
    row_block_size: int = 32,
) -> Lfm2PreparedJointAttention:
    """Prepare invariant fixed-A attention V/O work once."""

    verify_lfm2_cached_second_order_evidence(value_evidence)
    verify_lfm2_cached_second_order_evidence(output_evidence)
    verify_lfm2_attention_bridge(attention_bridge)
    value_source, value_source_sha = _source_weight(
        value_weight,
        "value_weight",
    )
    output_source, output_source_sha = _source_weight(
        output_weight,
        "output_weight",
    )
    if value_source.shape != (
        value_evidence.output_features,
        value_evidence.input_features,
    ):
        raise Lfm2CachedSecondOrderError(
            "Value weight geometry does not match its evidence."
        )
    if output_source.shape != (
        output_evidence.output_features,
        output_evidence.input_features,
    ):
        raise Lfm2CachedSecondOrderError(
            "Output weight geometry does not match its evidence."
        )
    if (
        value_evidence.sample_count
        != attention_bridge.batch_size * attention_bridge.key_tokens
        or output_evidence.sample_count
        != attention_bridge.batch_size * attention_bridge.query_tokens
        or value_evidence.output_features != attention_bridge.value_hidden_size
        or output_evidence.input_features != attention_bridge.query_hidden_size
    ):
        raise Lfm2CachedSecondOrderError(
            "Joint V/O evidence does not match the sealed attention/GQA "
            "bridge geometry."
        )
    reproduced_child_context = _attention_context_from_values(
        value_evidence.child_outputs,
        attention_bridge,
    )
    if not torch.allclose(
        reproduced_child_context,
        output_evidence.child_inputs,
        rtol=5e-3,
        atol=5e-4,
    ):
        maximum_error = float(
            (reproduced_child_context - output_evidence.child_inputs).abs().amax()
        )
        raise Lfm2CachedSecondOrderError(
            "Sealed attention probabilities do not reproduce the cached "
            f"child O input (maximum error {maximum_error:.8g})."
        )
    for name, value in (
        ("value_name", value_name),
        ("output_name", output_name),
        ("value_role", value_role),
        ("output_role", output_role),
    ):
        if not isinstance(value, str) or not value.strip():
            raise Lfm2CachedSecondOrderError(f"{name} must be a nonempty string.")
    objective_weight_ppm = _ppm(
        value_objective_weight,
        "value_objective_weight",
        minimum=0.0,
        maximum=10.0,
    )
    mismatch_ppm = _mismatch_ppm_values(mismatch_coefficients)
    value_statistics = compute_lfm2_second_order_statistics(
        value_evidence,
        damping_ratio=damping_ratio,
        parent_input_blend=parent_input_blend,
        output_damping_ratio=output_damping_ratio,
    )
    output_statistics = compute_lfm2_second_order_statistics(
        output_evidence,
        damping_ratio=damping_ratio,
        parent_input_blend=parent_input_blend,
        output_damping_ratio=output_damping_ratio,
    )
    value_input_inverse_cholesky = _inverse_cholesky(
        value_statistics.h_in
    )
    value_input_inverse_cholesky_sha = _tensor_fingerprint(
        value_input_inverse_cholesky
    )
    value_output_inverse_blocks = _output_inverse_blocks(
        value_statistics.h_out,
        rows=int(value_source.shape[0]),
        row_block_size=row_block_size,
    )
    current_value = _candidate_objective(
        value_source,
        value_source,
        value_evidence,
        value_statistics.h_out,
    )
    current_output = _candidate_objective(
        output_source,
        output_source,
        output_evidence,
        output_statistics.h_out,
    )
    current_joint = current_output + (objective_weight_ppm / 1_000_000 * current_value)
    targets = []
    for coefficient_ppm in mismatch_ppm:
        target = _corrected_target(
            value_source,
            value_evidence,
            value_statistics,
            mismatch_coefficient=coefficient_ppm / 1_000_000,
            row_block_size=row_block_size,
        )
        targets.append(
            Lfm2PreparedMismatchTarget(
                mismatch_coefficient_ppm=coefficient_ppm,
                target_weight=target,
                full_precision_objective=_candidate_objective(
                    value_source,
                    target,
                    value_evidence,
                    value_statistics.h_out,
                ),
                target_sha256=_tensor_fingerprint(target),
            )
        )
    evidence_bundle_sha = _sha256_canonical(
        {
            "value": value_evidence.evidence_sha256,
            "output": output_evidence.evidence_sha256,
            "attention_bridge": attention_bridge.bridge_sha256,
        }
    )
    source_bundle_sha = _sha256_canonical(
        {"value": value_source_sha, "output": output_source_sha}
    )
    prepared_value_sha = _tensor_fingerprint(value_source)
    prepared_output_sha = _tensor_fingerprint(output_source)
    audit = Lfm2PreparedComputationAudit(
        invariant_statistics_build_count=2,
        invariant_target_build_count=len(targets),
        invariant_input_factorization_build_count=1,
        invariant_output_factorization_build_count=len(
            value_output_inverse_blocks
        ),
        candidate_dependent_statistics_per_option=2,
        candidate_dependent_targets_per_option=2,
    )
    provisional = Lfm2PreparedJointAttention(
        schema=_PREPARED_JOINT_SCHEMA,
        value_name=value_name.strip(),
        output_name=output_name.strip(),
        value_role=value_role.strip(),
        output_role=output_role.strip(),
        value_source=value_source,
        output_source=output_source,
        value_source_sha256=value_source_sha,
        output_source_sha256=output_source_sha,
        prepared_value_source_sha256=prepared_value_sha,
        prepared_output_source_sha256=prepared_output_sha,
        value_evidence=value_evidence,
        output_evidence=output_evidence,
        attention_bridge=attention_bridge,
        value_statistics=value_statistics,
        output_baseline_statistics=output_statistics,
        value_input_inverse_cholesky=value_input_inverse_cholesky,
        value_input_inverse_cholesky_sha256=(
            value_input_inverse_cholesky_sha
        ),
        value_output_inverse_cholesky_blocks=(
            value_output_inverse_blocks
        ),
        value_targets=tuple(targets),
        current_joint_objective=current_joint,
        value_objective_weight_ppm=objective_weight_ppm,
        damping_ratio=float(damping_ratio),
        parent_input_blend=float(parent_input_blend),
        output_damping_ratio=float(output_damping_ratio),
        row_block_size=row_block_size,
        evidence_bundle_sha256=evidence_bundle_sha,
        source_bundle_sha256=source_bundle_sha,
        prepared_sha256="",
        computation_audit=audit,
    )
    return Lfm2PreparedJointAttention(
        schema=provisional.schema,
        value_name=provisional.value_name,
        output_name=provisional.output_name,
        value_role=provisional.value_role,
        output_role=provisional.output_role,
        value_source=value_source,
        output_source=output_source,
        value_source_sha256=value_source_sha,
        output_source_sha256=output_source_sha,
        prepared_value_source_sha256=prepared_value_sha,
        prepared_output_source_sha256=prepared_output_sha,
        value_evidence=value_evidence,
        output_evidence=output_evidence,
        attention_bridge=attention_bridge,
        value_statistics=value_statistics,
        output_baseline_statistics=output_statistics,
        value_input_inverse_cholesky=value_input_inverse_cholesky,
        value_input_inverse_cholesky_sha256=(
            value_input_inverse_cholesky_sha
        ),
        value_output_inverse_cholesky_blocks=(
            value_output_inverse_blocks
        ),
        value_targets=tuple(targets),
        current_joint_objective=current_joint,
        value_objective_weight_ppm=objective_weight_ppm,
        damping_ratio=float(damping_ratio),
        parent_input_blend=float(parent_input_blend),
        output_damping_ratio=float(output_damping_ratio),
        row_block_size=row_block_size,
        evidence_bundle_sha256=evidence_bundle_sha,
        source_bundle_sha256=source_bundle_sha,
        prepared_sha256=_sha256_canonical(_prepared_joint_body(provisional, targets)),
        computation_audit=audit,
    )


def verify_lfm2_prepared_joint_attention(
    prepared: Lfm2PreparedJointAttention,
) -> None:
    """Fail closed if prepared joint-attention state changed."""

    if not isinstance(prepared, Lfm2PreparedJointAttention):
        raise Lfm2CachedSecondOrderError("prepared must be Lfm2PreparedJointAttention.")
    if prepared.schema != _PREPARED_JOINT_SCHEMA:
        raise Lfm2CachedSecondOrderError("Unsupported prepared joint-attention schema.")
    verify_lfm2_cached_second_order_evidence(prepared.value_evidence)
    verify_lfm2_cached_second_order_evidence(prepared.output_evidence)
    verify_lfm2_attention_bridge(prepared.attention_bridge)
    _verify_statistics(
        prepared.value_statistics,
        prepared.value_evidence,
    )
    _verify_statistics(
        prepared.output_baseline_statistics,
        prepared.output_evidence,
    )
    if (
        _tensor_fingerprint(prepared.value_source)
        != prepared.prepared_value_source_sha256
        or _tensor_fingerprint(prepared.output_source)
        != prepared.prepared_output_source_sha256
    ):
        raise Lfm2CachedSecondOrderError("Prepared joint source authentication failed.")
    if (
        prepared.value_input_inverse_cholesky.shape
        != prepared.value_statistics.h_in.shape
        or prepared.value_input_inverse_cholesky.dtype != torch.float64
        or prepared.value_input_inverse_cholesky.device.type != "cpu"
        or not bool(
            torch.isfinite(
                prepared.value_input_inverse_cholesky
            ).all()
        )
        or bool(
            torch.diag(prepared.value_input_inverse_cholesky).le(0).any()
        )
        or _tensor_fingerprint(prepared.value_input_inverse_cholesky)
        != prepared.value_input_inverse_cholesky_sha256
    ):
        raise Lfm2CachedSecondOrderError(
            "Prepared joint input inverse-Cholesky authentication failed."
        )
    _verify_inverse_blocks(
        prepared.value_output_inverse_cholesky_blocks,
        rows=int(prepared.value_source.shape[0]),
        row_block_size=prepared.row_block_size,
        expected=prepared.value_statistics.h_out is not None,
    )
    targets = _target_map(prepared.value_targets)
    for target in targets.values():
        if (
            target.target_weight.shape != prepared.value_source.shape
            or _tensor_fingerprint(target.target_weight) != target.target_sha256
        ):
            raise Lfm2CachedSecondOrderError(
                "Prepared joint target authentication failed."
            )
    if (
        _sha256_canonical(_prepared_joint_body(prepared, prepared.value_targets))
        != prepared.prepared_sha256
    ):
        raise Lfm2CachedSecondOrderError(
            "Prepared joint metadata authentication failed."
        )


def solve_lfm2_prepared_joint_attention_option(
    prepared: Lfm2PreparedJointAttention,
    *,
    bit_width: int,
    group_size: int,
    clipping_ratio: float,
    rounding_offset: float,
    outliers_per_group: int,
    mismatch_coefficient: float,
    propagation_coefficient: float,
) -> Lfm2JointAttentionOption:
    """Encode one fixed-A V/O grid point from prepared invariant state."""

    verify_lfm2_prepared_joint_attention(prepared)
    quantizers = _grid_configs(
        input_features=prepared.value_evidence.input_features,
        second_input_features=prepared.output_evidence.input_features,
        bit_widths=(bit_width,),
        group_sizes=(group_size,),
        clipping_ratios=(clipping_ratio,),
        rounding_offsets=(rounding_offset,),
        outliers_per_group=(outliers_per_group,),
        mismatch_coefficients=(mismatch_coefficient,),
    )
    if len(quantizers) != 1:
        raise Lfm2CachedSecondOrderError(
            "A prepared joint option requires one quantizer point."
        )
    quantizer = quantizers[0]
    propagation_ppm = _ppm(
        propagation_coefficient,
        "propagation_coefficient",
        minimum=0.0,
        maximum=2.0,
    )
    config = Lfm2JointAttentionConfig(
        quantizer=quantizer,
        propagation_coefficient_ppm=propagation_ppm,
        value_objective_weight_ppm=prepared.value_objective_weight_ppm,
    )
    try:
        target = _target_map(prepared.value_targets)[
            quantizer.mismatch_coefficient_ppm
        ].target_weight
    except KeyError as exc:
        raise Lfm2CachedSecondOrderError(
            "The requested joint mismatch target was not prepared."
        ) from exc
    branch_arguments = {
        "value_source": prepared.value_source,
        "output_source": prepared.output_source,
        "value_target": target,
        "value_statistics": prepared.value_statistics,
        "value_evidence": prepared.value_evidence,
        "output_evidence": prepared.output_evidence,
        "attention_bridge": prepared.attention_bridge,
        "config": config,
        "value_name": prepared.value_name,
        "value_role": prepared.value_role,
        "output_name": prepared.output_name,
        "output_role": prepared.output_role,
        "value_source_sha": prepared.value_source_sha256,
        "output_source_sha": prepared.output_source_sha256,
        "damping_ratio": prepared.damping_ratio,
        "parent_input_blend": prepared.parent_input_blend,
        "output_damping_ratio": prepared.output_damping_ratio,
        "row_block_size": prepared.row_block_size,
    }
    rtn = _joint_branch(successive=False, **branch_arguments)
    successive = _joint_branch(successive=True, **branch_arguments)
    tolerance = 1e-12 * max(1.0, abs(rtn[3]))
    if successive[3] <= rtn[3] + tolerance:
        selected = successive
        selected_method = "successive"
    else:
        selected = rtn
        selected_method = "rtn_rollback"
    config_sha = _configuration_sha256(config)
    physical_bundle_sha = _sha256_canonical(
        {
            "value": selected[0].physical_payload_sha256,
            "output": selected[1].physical_payload_sha256,
            "config_sha256": config_sha,
        }
    )
    option_id = (
        f"{prepared.value_name}+{prepared.output_name}:"
        f"{quantizer.option_suffix}:p{propagation_ppm:+d}:"
        f"vw{prepared.value_objective_weight_ppm}"
    )
    option_sha = _sha256_canonical(
        {
            "option_id": option_id,
            "evidence_bundle_sha256": prepared.evidence_bundle_sha256,
            "physical_bundle_sha256": physical_bundle_sha,
            "selected_method": selected_method,
            "selected_joint_objective": selected[3],
        }
    )
    total_physical = (
        selected[0].physical_record_bytes + selected[1].physical_record_bytes
    )
    total_dense = selected[0].dense_bf16_bytes + selected[1].dense_bf16_bytes
    output_selected = selected[3] - config.value_objective_weight * selected[2]
    audit = Lfm2SecondOrderOpsAudit(
        output_hessian_row_compensation=(
            prepared.value_statistics.h_out is not None
            or prepared.output_baseline_statistics.h_out is not None
        ),
        fixed_attention_probability_bridge=True,
    )
    result = Lfm2JointAttentionOption(
        option_id=option_id,
        config=config,
        value_payload=selected[0],
        output_payload=selected[1],
        selected_method=selected_method,
        current_joint_objective=prepared.current_joint_objective,
        rtn_joint_objective=rtn[3],
        raw_successive_joint_objective=successive[3],
        selected_joint_objective=selected[3],
        value_selected_objective=selected[2],
        output_selected_objective=output_selected,
        total_physical_record_bytes=total_physical,
        total_dense_bf16_bytes=total_dense,
        total_physical_byte_savings=total_dense - total_physical,
        evidence_bundle_sha256=prepared.evidence_bundle_sha256,
        config_sha256=config_sha,
        physical_bundle_sha256=physical_bundle_sha,
        option_sha256=option_sha,
        ops_audit=audit,
    )
    verify_lfm2_prepared_joint_attention(prepared)
    return result


__all__ = [
    "Lfm2PreparedComputationAudit",
    "Lfm2PreparedJointAttention",
    "Lfm2PreparedMismatchTarget",
    "Lfm2PreparedSecondOrderSite",
    "prepare_lfm2_joint_attention_vo",
    "prepare_lfm2_second_order_site",
    "solve_lfm2_prepared_joint_attention_option",
    "solve_lfm2_prepared_second_order_option",
    "verify_lfm2_prepared_joint_attention",
    "verify_lfm2_prepared_second_order_site",
]
