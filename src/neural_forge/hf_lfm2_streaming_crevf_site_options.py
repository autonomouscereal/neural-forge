"""Memory-bounded CREVF site frontiers from cached second-order solves.

The corrected second-order solver intentionally returns a reconstructed tensor
so one candidate can be audited.  Retaining an entire model-scale grid in that
form would be wasteful.  This module invokes one grid point at a time, converts
it immediately to authenticated CREVF bytes plus integer loss, and keeps only
top-K compact bundles in each physical-byte band.

No model, callback, optimizer, or trace producer is accepted here.  Query/key
and exact norm/control precision floors are applied before a solver invocation
and are independently rechecked by :class:`CrevfSite`.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP, localcontext
from typing import Any, Iterable, Sequence

import torch
from torch import Tensor, nn

from .hf_lfm2_cached_second_order_attention_solver import (
    Lfm2CachedSecondOrderEvidence,
    Lfm2SealedAttentionBridge,
    verify_lfm2_attention_bridge,
    verify_lfm2_cached_second_order_evidence,
)
from .hf_lfm2_causal_residual_entropy_value_fold import (
    build_crevf_exact_spec,
    encode_crevf_tensor,
)
from .hf_lfm2_crevf_budget import (
    CrevfSite,
    CrevfSiteOption,
    role_minimum_integer_width,
    semantic_minimum_integer_width,
    tensor_identity_minimum_width,
)
from .hf_lfm2_prepared_second_order_site import (
    prepare_lfm2_joint_attention_vo,
    prepare_lfm2_second_order_site,
    solve_lfm2_prepared_joint_attention_option,
    solve_lfm2_prepared_second_order_option,
)


_STREAM_SCHEMA = "neural-forge.lfm2.streaming-crevf-site.v1"


class Lfm2StreamingCrevfError(ValueError):
    """A streaming frontier request or its retained result is invalid."""


@dataclass(frozen=True, slots=True)
class Lfm2StreamingCrevfConfig:
    """Grid axes, memory frontier, and deterministic loss-unit policy."""

    bit_widths: tuple[int, ...] = (4, 5, 6, 7, 8)
    group_sizes: tuple[int, ...] = (32, 64, 128, 256)
    clipping_ratios: tuple[float, ...] = (0.9, 1.0)
    rounding_offsets: tuple[float, ...] = (0.0,)
    outliers_per_group: tuple[int, ...] = (0,)
    mismatch_coefficients: tuple[float, ...] = (0.0, 1.0)
    propagation_coefficients: tuple[float, ...] = (0.5, 1.0)
    top_k_per_physical_band: int = 2
    physical_byte_band_width: int = 1_048_576
    objective_loss_scale: int = 1_000_000_000_000
    damping_ratio: float = 1e-4
    parent_input_blend: float = 0.5
    output_damping_ratio: float = 1e-4
    row_block_size: int = 32
    max_grid_points: int = 4096

    def __post_init__(self) -> None:
        for name in (
            "bit_widths",
            "group_sizes",
            "clipping_ratios",
            "rounding_offsets",
            "outliers_per_group",
            "mismatch_coefficients",
            "propagation_coefficients",
        ):
            values = tuple(getattr(self, name))
            if not values or len(values) != len(set(values)):
                raise Lfm2StreamingCrevfError(
                    f"{name} must be nonempty and contain no duplicates."
                )
            object.__setattr__(self, name, values)
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value not in range(4, 9)
            for value in self.bit_widths
        ):
            raise Lfm2StreamingCrevfError(
                "bit_widths must contain only integers from four through eight."
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value not in {32, 64, 128, 256}
            for value in self.group_sizes
        ):
            raise Lfm2StreamingCrevfError(
                "group_sizes must contain only 32, 64, 128, or 256."
            )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.outliers_per_group
        ):
            raise Lfm2StreamingCrevfError(
                "outliers_per_group must contain nonnegative integers."
            )
        if any(value >= min(self.group_sizes) for value in self.outliers_per_group):
            raise Lfm2StreamingCrevfError(
                "Each outlier count must be smaller than every group size."
            )

        def validate_ppm_axis(
            values: Sequence[float],
            name: str,
            minimum: float,
            maximum: float,
        ) -> None:
            for value in values:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise Lfm2StreamingCrevfError(
                        f"{name} must contain numeric values."
                    )
                number = float(value)
                ppm = int(round(number * 1_000_000))
                if (
                    not math.isfinite(number)
                    or not minimum <= number <= maximum
                    or not math.isclose(
                        number,
                        ppm / 1_000_000,
                        abs_tol=1e-12,
                    )
                ):
                    raise Lfm2StreamingCrevfError(
                        f"{name} values must be finite, in range, and exactly "
                        "representable in parts per million."
                    )

        validate_ppm_axis(
            self.clipping_ratios,
            "clipping_ratios",
            0.25,
            1.0,
        )
        validate_ppm_axis(
            self.rounding_offsets,
            "rounding_offsets",
            -0.49,
            0.49,
        )
        validate_ppm_axis(
            self.mismatch_coefficients,
            "mismatch_coefficients",
            -2.0,
            2.0,
        )
        validate_ppm_axis(
            self.propagation_coefficients,
            "propagation_coefficients",
            0.0,
            2.0,
        )
        for value, name in (
            (self.damping_ratio, "damping_ratio"),
            (self.parent_input_blend, "parent_input_blend"),
            (self.output_damping_ratio, "output_damping_ratio"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise Lfm2StreamingCrevfError(f"{name} must be finite and in [0, 1].")
        for name in (
            "top_k_per_physical_band",
            "physical_byte_band_width",
            "objective_loss_scale",
            "row_block_size",
            "max_grid_points",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise Lfm2StreamingCrevfError(f"{name} must be a positive integer.")


@dataclass(frozen=True, slots=True)
class Lfm2StreamingTensorSiteRequest:
    """One ordinary tensor site backed by sealed cached evidence."""

    name: str
    role: str
    weight: Tensor = field(repr=False, compare=False)
    evidence: Lfm2CachedSecondOrderEvidence = field(repr=False, compare=False)
    semantic_kind: str = "other"
    importance_ppm: int = 1_000_000


@dataclass(frozen=True, slots=True)
class Lfm2StreamingJointAttentionRequest:
    """One atomic attention V/O site with a sealed fixed-A GQA bridge."""

    value_name: str
    output_name: str
    value_weight: Tensor = field(repr=False, compare=False)
    output_weight: Tensor = field(repr=False, compare=False)
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
    value_role: str = "attention.v"
    output_role: str = "attention.o"
    semantic_kind: str = "other"
    importance_ppm: int = 1_000_000
    value_objective_weight: float = 0.1


@dataclass(frozen=True, slots=True)
class Lfm2RetainedCrevfOption:
    """Tensor-free provenance for one retained authenticated bundle."""

    option_id: str
    solver_option_id: str
    band_index: int
    selected_method: str
    selected_objective: str
    loss_units: int
    encoded_record_bytes: int
    persistent_encoded_bytes: int
    minimum_bit_width: int
    exact_bf16: bool
    bundle_sha256: str
    solver_option_sha256: str


@dataclass(frozen=True, slots=True)
class Lfm2PhysicalByteBandReport:
    """Generation and retention counts for one exact byte interval."""

    band_index: int
    minimum_physical_bytes: int
    maximum_physical_bytes: int
    generated_option_count: int
    retained_option_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Lfm2StreamingCrevfOpsAudit:
    """Bounded-memory and zero-execution audit for one site build."""

    solver_invocations: int
    generated_encoded_options: int
    retained_encoded_options: int
    discarded_encoded_options: int
    returned_dense_tensor_count: int
    invariant_statistics_build_count: int
    invariant_target_build_count: int
    candidate_dependent_statistics_build_count: int
    candidate_dependent_target_build_count: int
    parent_model_calls: int = 0
    child_model_calls: int = 0
    forward_calls: int = 0
    generation_calls: int = 0
    gradient_calls: int = 0
    optimizer_steps: int = 0
    training_steps: int = 0


@dataclass(frozen=True, slots=True)
class Lfm2StreamingCrevfSiteBuild:
    """A compact allocator-ready site plus authenticated build provenance."""

    schema: str
    site: CrevfSite
    retained_options: tuple[Lfm2RetainedCrevfOption, ...]
    physical_bands: tuple[Lfm2PhysicalByteBandReport, ...]
    source_bundle_sha256: str
    evidence_bundle_sha256: str
    config_sha256: str
    build_sha256: str
    ops_audit: Lfm2StreamingCrevfOpsAudit


@dataclass(frozen=True, slots=True)
class _CompactCandidate:
    option: CrevfSiteOption
    provenance: Lfm2RetainedCrevfOption

    @property
    def rank_key(self) -> tuple[Any, ...]:
        return (
            self.option.loss_units,
            self.option.encoded_record_bytes,
            self.option.persistent_encoded_bytes,
            self.option.option_id,
            self.option.bundle_sha256,
        )


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _tensor_raw_bytes(value: Tensor) -> bytes:
    return (
        value.detach()
        .to(device="cpu")
        .contiguous()
        .view(torch.uint8)
        .numpy()
        .tobytes(order="C")
    )


def _tensor_sha256(value: Tensor) -> str:
    header = _canonical_bytes(
        {
            "dtype": str(value.dtype),
            "shape": tuple(int(part) for part in value.shape),
        }
    )
    return hashlib.sha256(header + b"\0" + _tensor_raw_bytes(value)).hexdigest()


def _source_tensor(value: Any, name: str) -> tuple[Tensor, str]:
    if not isinstance(value, Tensor):
        raise Lfm2StreamingCrevfError(f"{name} must be a Tensor.")
    if isinstance(value, nn.Parameter):
        raise Lfm2StreamingCrevfError(f"{name} cannot be an nn.Parameter.")
    if (
        value.requires_grad
        or value.device.type == "meta"
        or value.dtype != torch.bfloat16
        or value.numel() == 0
        or not bool(torch.isfinite(value).all())
    ):
        raise Lfm2StreamingCrevfError(
            f"{name} must be finite materialized BF16 without gradients."
        )
    return value, _tensor_sha256(value)


def _importance_ppm(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2StreamingCrevfError("importance_ppm must be a positive integer.")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Lfm2StreamingCrevfError(f"{name} must be nonempty text.")
    return value.strip()


def _float_text(value: float) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise Lfm2StreamingCrevfError("Grid floats must be finite.")
    return format(number, ".17g")


def _config_body(config: Lfm2StreamingCrevfConfig) -> dict[str, Any]:
    return {
        "bit_widths": config.bit_widths,
        "group_sizes": config.group_sizes,
        "clipping_ratios": tuple(
            _float_text(value) for value in config.clipping_ratios
        ),
        "rounding_offsets": tuple(
            _float_text(value) for value in config.rounding_offsets
        ),
        "outliers_per_group": config.outliers_per_group,
        "mismatch_coefficients": tuple(
            _float_text(value) for value in config.mismatch_coefficients
        ),
        "propagation_coefficients": tuple(
            _float_text(value) for value in config.propagation_coefficients
        ),
        "top_k_per_physical_band": config.top_k_per_physical_band,
        "physical_byte_band_width": config.physical_byte_band_width,
        "objective_loss_scale": config.objective_loss_scale,
        "damping_ratio": _float_text(config.damping_ratio),
        "parent_input_blend": _float_text(config.parent_input_blend),
        "output_damping_ratio": _float_text(config.output_damping_ratio),
        "row_block_size": config.row_block_size,
        "max_grid_points": config.max_grid_points,
    }


def _loss_units(
    objective: float,
    *,
    scale: int,
    importance_ppm: int,
) -> int:
    number = float(objective)
    if not math.isfinite(number) or number < 0:
        raise Lfm2StreamingCrevfError(
            "A selected objective must be finite and nonnegative."
        )
    with localcontext() as context:
        context.prec = 80
        scaled = (
            Decimal(_float_text(number))
            * Decimal(scale)
            * Decimal(importance_ppm)
            / Decimal(1_000_000)
        )
        return int(scaled.to_integral_value(rounding=ROUND_HALF_UP))


def _precision_floor(name: str, role: str, semantic_kind: str) -> int:
    declared_floor = role_minimum_integer_width(role)
    identity_floor = tensor_identity_minimum_width(name)
    if declared_floor < identity_floor:
        raise Lfm2StreamingCrevfError(
            f"Role {role!r} understates the sealed {identity_floor}-bit "
            f"identity floor for {name!r}."
        )
    return max(
        declared_floor,
        identity_floor,
        semantic_minimum_integer_width(semantic_kind),
    )


def _effective_bits(
    config: Lfm2StreamingCrevfConfig,
    minimum_width: int,
) -> tuple[int, ...]:
    values = tuple(
        sorted(
            value
            for value in config.bit_widths
            if isinstance(value, int)
            and not isinstance(value, bool)
            and value >= minimum_width
        )
    )
    if not values:
        raise Lfm2StreamingCrevfError(
            f"No requested bit width satisfies the hard {minimum_width}-bit floor."
        )
    return values


def _effective_groups(
    config: Lfm2StreamingCrevfConfig,
    input_features: Sequence[int],
) -> tuple[int, ...]:
    groups = tuple(
        sorted(
            value
            for value in config.group_sizes
            if isinstance(value, int)
            and not isinstance(value, bool)
            and all(features % value == 0 for features in input_features)
        )
    )
    if not groups:
        raise Lfm2StreamingCrevfError(
            "No requested group size is row-aligned for this site."
        )
    return groups


def _grid_point_count(
    axes: Iterable[Sequence[Any]],
    *,
    maximum: int,
) -> int:
    count = math.prod(len(tuple(axis)) for axis in axes)
    if count <= 0 or count > maximum:
        raise Lfm2StreamingCrevfError(
            f"Streaming grid has {count} points; max_grid_points is {maximum}."
        )
    return count


def _band_index(encoded_record_bytes: int, width: int) -> int:
    if encoded_record_bytes <= 0:
        raise Lfm2StreamingCrevfError(
            "Encoded CREVF records must occupy physical bytes."
        )
    return (encoded_record_bytes - 1) // width


def _retain_candidate(
    retained: dict[int, dict[str, _CompactCandidate]],
    generated_per_band: dict[int, int],
    candidate: _CompactCandidate,
    config: Lfm2StreamingCrevfConfig,
) -> None:
    band = candidate.provenance.band_index
    generated_per_band[band] = generated_per_band.get(band, 0) + 1
    choices = retained.setdefault(band, {})
    current = choices.get(candidate.option.bundle_sha256)
    if current is None or candidate.rank_key < current.rank_key:
        choices[candidate.option.bundle_sha256] = candidate
    ranked = sorted(choices.values(), key=lambda value: value.rank_key)
    choices.clear()
    choices.update(
        (value.option.bundle_sha256, value)
        for value in ranked[: config.top_k_per_physical_band]
    )


def _compact_option(
    *,
    solver_option_id: str,
    solver_option_sha256: str,
    selected_method: str,
    selected_objective: float,
    tensor_blob: bytes,
    correction_tensor_blobs: tuple[bytes, ...],
    config: Lfm2StreamingCrevfConfig,
    importance_ppm: int,
) -> _CompactCandidate:
    loss = _loss_units(
        selected_objective,
        scale=config.objective_loss_scale,
        importance_ppm=importance_ppm,
    )
    option = CrevfSiteOption(
        option_id=solver_option_id,
        tensor_blob=tensor_blob,
        loss_units=loss,
        correction_tensor_blobs=correction_tensor_blobs,
    )
    band = _band_index(
        option.encoded_record_bytes,
        config.physical_byte_band_width,
    )
    provenance = Lfm2RetainedCrevfOption(
        option_id=option.option_id,
        solver_option_id=solver_option_id,
        band_index=band,
        selected_method=selected_method,
        selected_objective=_float_text(selected_objective),
        loss_units=loss,
        encoded_record_bytes=option.encoded_record_bytes,
        persistent_encoded_bytes=option.persistent_encoded_bytes,
        minimum_bit_width=option.minimum_bit_width,
        exact_bf16=option.exact_bf16,
        bundle_sha256=option.bundle_sha256,
        solver_option_sha256=solver_option_sha256,
    )
    return _CompactCandidate(option=option, provenance=provenance)


def _exact_candidate(
    *,
    name: str,
    role: str,
    values: Tensor,
    correction: tuple[tuple[str, str, Tensor], ...],
    config: Lfm2StreamingCrevfConfig,
) -> _CompactCandidate:
    blob = encode_crevf_tensor(
        build_crevf_exact_spec(name=name, role=role, values=values)
    )
    corrections = tuple(
        encode_crevf_tensor(
            build_crevf_exact_spec(
                name=correction_name,
                role=correction_role,
                values=correction_values,
            )
        )
        for correction_name, correction_role, correction_values in correction
    )
    option_id = f"{name}:exact-bf16"
    option = CrevfSiteOption(
        option_id=option_id,
        tensor_blob=blob,
        loss_units=0,
        correction_tensor_blobs=corrections,
    )
    band = _band_index(
        option.encoded_record_bytes,
        config.physical_byte_band_width,
    )
    seal = _canonical_sha256(
        {
            "option_id": option_id,
            "bundle_sha256": option.bundle_sha256,
            "method": "exact_bf16",
        }
    )
    return _CompactCandidate(
        option=option,
        provenance=Lfm2RetainedCrevfOption(
            option_id=option_id,
            solver_option_id=option_id,
            band_index=band,
            selected_method="exact_bf16",
            selected_objective="0",
            loss_units=0,
            encoded_record_bytes=option.encoded_record_bytes,
            persistent_encoded_bytes=option.persistent_encoded_bytes,
            minimum_bit_width=option.minimum_bit_width,
            exact_bf16=True,
            bundle_sha256=option.bundle_sha256,
            solver_option_sha256=seal,
        ),
    )


def _finalize(
    *,
    site_id: str,
    role: str,
    semantic_kind: str,
    retained: dict[int, dict[str, _CompactCandidate]],
    generated_per_band: dict[int, int],
    generated_count: int,
    solver_invocations: int,
    invariant_statistics_build_count: int,
    invariant_target_build_count: int,
    candidate_dependent_statistics_build_count: int,
    candidate_dependent_target_build_count: int,
    source_bundle_sha256: str,
    evidence_bundle_sha256: str,
    config: Lfm2StreamingCrevfConfig,
) -> Lfm2StreamingCrevfSiteBuild:
    candidates = tuple(
        sorted(
            (
                candidate
                for values in retained.values()
                for candidate in values.values()
            ),
            key=lambda value: value.option.option_id,
        )
    )
    if not candidates:
        raise Lfm2StreamingCrevfError(
            "The streaming frontier retained no encoded option."
        )
    options = tuple(candidate.option for candidate in candidates)
    site = CrevfSite(
        site_id=site_id,
        role=role,
        options=options,
        semantic_kind=semantic_kind,
    )
    _ = site.eligible_options
    provenance = tuple(candidate.provenance for candidate in candidates)
    reports = tuple(
        Lfm2PhysicalByteBandReport(
            band_index=index,
            minimum_physical_bytes=(index * config.physical_byte_band_width + 1),
            maximum_physical_bytes=((index + 1) * config.physical_byte_band_width),
            generated_option_count=generated_per_band[index],
            retained_option_ids=tuple(
                sorted(value.option.option_id for value in retained[index].values())
            ),
        )
        for index in sorted(retained)
    )
    config_sha = _canonical_sha256(_config_body(config))
    body = {
        "schema": _STREAM_SCHEMA,
        "site_id": site.site_id,
        "role": site.role,
        "semantic_kind": site.semantic_kind,
        "source_bundle_sha256": source_bundle_sha256,
        "evidence_bundle_sha256": evidence_bundle_sha256,
        "config_sha256": config_sha,
        "retained_options": tuple(
            {
                "option_id": value.option_id,
                "band_index": value.band_index,
                "loss_units": value.loss_units,
                "encoded_record_bytes": value.encoded_record_bytes,
                "bundle_sha256": value.bundle_sha256,
                "solver_option_sha256": value.solver_option_sha256,
            }
            for value in provenance
        ),
    }
    audit = Lfm2StreamingCrevfOpsAudit(
        solver_invocations=solver_invocations,
        generated_encoded_options=generated_count,
        retained_encoded_options=len(options),
        discarded_encoded_options=generated_count - len(options),
        returned_dense_tensor_count=0,
        invariant_statistics_build_count=(invariant_statistics_build_count),
        invariant_target_build_count=invariant_target_build_count,
        candidate_dependent_statistics_build_count=(
            candidate_dependent_statistics_build_count
        ),
        candidate_dependent_target_build_count=(candidate_dependent_target_build_count),
    )
    return Lfm2StreamingCrevfSiteBuild(
        schema=_STREAM_SCHEMA,
        site=site,
        retained_options=provenance,
        physical_bands=reports,
        source_bundle_sha256=source_bundle_sha256,
        evidence_bundle_sha256=evidence_bundle_sha256,
        config_sha256=config_sha,
        build_sha256=_canonical_sha256(body),
        ops_audit=audit,
    )


def build_lfm2_streaming_crevf_site(
    request: Lfm2StreamingTensorSiteRequest,
    *,
    config: Lfm2StreamingCrevfConfig = Lfm2StreamingCrevfConfig(),
) -> Lfm2StreamingCrevfSiteBuild:
    """Build one allocator-ready site while retaining no dense candidates."""

    if not isinstance(request, Lfm2StreamingTensorSiteRequest):
        raise Lfm2StreamingCrevfError("request must be Lfm2StreamingTensorSiteRequest.")
    if not isinstance(config, Lfm2StreamingCrevfConfig):
        raise Lfm2StreamingCrevfError("config must be Lfm2StreamingCrevfConfig.")
    name = _text(request.name, "name")
    role = _text(request.role, "role")
    semantic_kind = _text(request.semantic_kind, "semantic_kind")
    importance = _importance_ppm(request.importance_ppm)
    weight, source_sha = _source_tensor(request.weight, "weight")
    verify_lfm2_cached_second_order_evidence(request.evidence)
    minimum_width = _precision_floor(name, role, semantic_kind)
    retained: dict[int, dict[str, _CompactCandidate]] = {}
    generated_per_band: dict[int, int] = {}
    solver_invocations = 0
    generated_count = 0
    invariant_statistics_build_count = 0
    invariant_target_build_count = 0
    candidate_statistics_build_count = 0
    candidate_target_build_count = 0
    if minimum_width == 16:
        candidate = _exact_candidate(
            name=name,
            role=role,
            values=weight,
            correction=(),
            config=config,
        )
        _retain_candidate(
            retained,
            generated_per_band,
            candidate,
            config,
        )
        generated_count = 1
    else:
        if weight.ndim != 2:
            raise Lfm2StreamingCrevfError(
                "Quantized cached second-order sites require a 2D weight."
            )
        bits = _effective_bits(config, minimum_width)
        groups = _effective_groups(config, (int(weight.shape[1]),))
        axes = (
            bits,
            groups,
            config.clipping_ratios,
            config.rounding_offsets,
            config.outliers_per_group,
            config.mismatch_coefficients,
        )
        expected = _grid_point_count(
            axes,
            maximum=config.max_grid_points,
        )
        prepared = prepare_lfm2_second_order_site(
            weight,
            request.evidence,
            name=name,
            role=role,
            mismatch_coefficients=config.mismatch_coefficients,
            damping_ratio=config.damping_ratio,
            parent_input_blend=config.parent_input_blend,
            output_damping_ratio=config.output_damping_ratio,
            row_block_size=config.row_block_size,
        )
        preparation_audit = prepared.computation_audit
        invariant_statistics_build_count = (
            preparation_audit.invariant_statistics_build_count
        )
        invariant_target_build_count = preparation_audit.invariant_target_build_count
        for (
            bit_width,
            group_size,
            clipping_ratio,
            rounding_offset,
            outlier_count,
            mismatch_coefficient,
        ) in itertools.product(*axes):
            selected = solve_lfm2_prepared_second_order_option(
                prepared,
                bit_width=bit_width,
                group_size=group_size,
                clipping_ratio=clipping_ratio,
                rounding_offset=rounding_offset,
                outliers_per_group=outlier_count,
                mismatch_coefficient=mismatch_coefficient,
            )
            solver_invocations += 1
            candidate = _compact_option(
                solver_option_id=selected.option_id,
                solver_option_sha256=selected.option_sha256,
                selected_method=selected.selected_method,
                selected_objective=selected.selected_objective,
                tensor_blob=selected.payload.tensor_blob,
                correction_tensor_blobs=(),
                config=config,
                importance_ppm=importance,
            )
            if (
                candidate.option.encoded_record_bytes
                != selected.payload.physical_record_bytes
            ):
                raise Lfm2StreamingCrevfError(
                    "Compact option bytes diverged from the solver payload."
                )
            _retain_candidate(
                retained,
                generated_per_band,
                candidate,
                config,
            )
            generated_count += 1
            del selected, candidate
        if generated_count != expected:
            raise Lfm2StreamingCrevfError(
                "Streaming grid enumeration diverged from its exact count."
            )
    if _tensor_sha256(weight) != source_sha:
        raise Lfm2StreamingCrevfError("Source tensor changed during streaming.")
    verify_lfm2_cached_second_order_evidence(request.evidence)
    return _finalize(
        site_id=name,
        role=role,
        semantic_kind=semantic_kind,
        retained=retained,
        generated_per_band=generated_per_band,
        generated_count=generated_count,
        solver_invocations=solver_invocations,
        invariant_statistics_build_count=(invariant_statistics_build_count),
        invariant_target_build_count=invariant_target_build_count,
        candidate_dependent_statistics_build_count=(candidate_statistics_build_count),
        candidate_dependent_target_build_count=(candidate_target_build_count),
        source_bundle_sha256=source_sha,
        evidence_bundle_sha256=request.evidence.evidence_sha256,
        config=config,
    )


def build_lfm2_streaming_joint_attention_site(
    request: Lfm2StreamingJointAttentionRequest,
    *,
    config: Lfm2StreamingCrevfConfig = Lfm2StreamingCrevfConfig(),
) -> Lfm2StreamingCrevfSiteBuild:
    """Build one top-K physical frontier of atomic attention V/O bundles."""

    if not isinstance(request, Lfm2StreamingJointAttentionRequest):
        raise Lfm2StreamingCrevfError(
            "request must be Lfm2StreamingJointAttentionRequest."
        )
    if not isinstance(config, Lfm2StreamingCrevfConfig):
        raise Lfm2StreamingCrevfError("config must be Lfm2StreamingCrevfConfig.")
    value_name = _text(request.value_name, "value_name")
    output_name = _text(request.output_name, "output_name")
    value_role = _text(request.value_role, "value_role")
    output_role = _text(request.output_role, "output_role")
    semantic_kind = _text(request.semantic_kind, "semantic_kind")
    importance = _importance_ppm(request.importance_ppm)
    if (
        isinstance(request.value_objective_weight, bool)
        or not isinstance(request.value_objective_weight, (int, float))
        or not math.isfinite(float(request.value_objective_weight))
        or not 0.0 <= float(request.value_objective_weight) <= 10.0
    ):
        raise Lfm2StreamingCrevfError(
            "value_objective_weight must be finite and in [0, 10]."
        )
    value_weight, value_source_sha = _source_tensor(
        request.value_weight,
        "value_weight",
    )
    output_weight, output_source_sha = _source_tensor(
        request.output_weight,
        "output_weight",
    )
    verify_lfm2_cached_second_order_evidence(request.value_evidence)
    verify_lfm2_cached_second_order_evidence(request.output_evidence)
    verify_lfm2_attention_bridge(request.attention_bridge)
    minimum_width = max(
        _precision_floor(value_name, value_role, semantic_kind),
        _precision_floor(output_name, output_role, "other"),
    )
    retained: dict[int, dict[str, _CompactCandidate]] = {}
    generated_per_band: dict[int, int] = {}
    solver_invocations = 0
    generated_count = 0
    invariant_statistics_build_count = 0
    invariant_target_build_count = 0
    candidate_statistics_build_count = 0
    candidate_target_build_count = 0
    if minimum_width == 16:
        candidate = _exact_candidate(
            name=value_name,
            role=value_role,
            values=value_weight,
            correction=((output_name, output_role, output_weight),),
            config=config,
        )
        _retain_candidate(
            retained,
            generated_per_band,
            candidate,
            config,
        )
        generated_count = 1
    else:
        if value_weight.ndim != 2 or output_weight.ndim != 2:
            raise Lfm2StreamingCrevfError(
                "Joint attention weights must both be two-dimensional."
            )
        bits = _effective_bits(config, minimum_width)
        groups = _effective_groups(
            config,
            (
                int(value_weight.shape[1]),
                int(output_weight.shape[1]),
            ),
        )
        axes = (
            bits,
            groups,
            config.clipping_ratios,
            config.rounding_offsets,
            config.outliers_per_group,
            config.mismatch_coefficients,
            config.propagation_coefficients,
        )
        expected = _grid_point_count(
            axes,
            maximum=config.max_grid_points,
        )
        prepared = prepare_lfm2_joint_attention_vo(
            value_weight,
            output_weight,
            request.value_evidence,
            request.output_evidence,
            attention_bridge=request.attention_bridge,
            value_name=value_name,
            output_name=output_name,
            value_role=value_role,
            output_role=output_role,
            mismatch_coefficients=config.mismatch_coefficients,
            value_objective_weight=request.value_objective_weight,
            damping_ratio=config.damping_ratio,
            parent_input_blend=config.parent_input_blend,
            output_damping_ratio=config.output_damping_ratio,
            row_block_size=config.row_block_size,
        )
        preparation_audit = prepared.computation_audit
        invariant_statistics_build_count = (
            preparation_audit.invariant_statistics_build_count
        )
        invariant_target_build_count = preparation_audit.invariant_target_build_count
        for (
            bit_width,
            group_size,
            clipping_ratio,
            rounding_offset,
            outlier_count,
            mismatch_coefficient,
            propagation_coefficient,
        ) in itertools.product(*axes):
            selected = solve_lfm2_prepared_joint_attention_option(
                prepared,
                bit_width=bit_width,
                group_size=group_size,
                clipping_ratio=clipping_ratio,
                rounding_offset=rounding_offset,
                outliers_per_group=outlier_count,
                mismatch_coefficient=mismatch_coefficient,
                propagation_coefficient=propagation_coefficient,
            )
            solver_invocations += 1
            candidate = _compact_option(
                solver_option_id=selected.option_id,
                solver_option_sha256=selected.option_sha256,
                selected_method=selected.selected_method,
                selected_objective=selected.selected_joint_objective,
                tensor_blob=selected.value_payload.tensor_blob,
                correction_tensor_blobs=(selected.output_payload.tensor_blob,),
                config=config,
                importance_ppm=importance,
            )
            if (
                candidate.option.encoded_record_bytes
                != selected.total_physical_record_bytes
            ):
                raise Lfm2StreamingCrevfError(
                    "Atomic V/O bytes diverged from the solver bundle."
                )
            _retain_candidate(
                retained,
                generated_per_band,
                candidate,
                config,
            )
            generated_count += 1
            del selected, candidate
        if generated_count != expected:
            raise Lfm2StreamingCrevfError(
                "Joint streaming enumeration diverged from its exact count."
            )
        candidate_statistics_build_count = (
            preparation_audit.candidate_dependent_statistics_per_option
            * solver_invocations
        )
        candidate_target_build_count = (
            preparation_audit.candidate_dependent_targets_per_option
            * solver_invocations
        )
    if (
        _tensor_sha256(value_weight) != value_source_sha
        or _tensor_sha256(output_weight) != output_source_sha
    ):
        raise Lfm2StreamingCrevfError(
            "An attention source tensor changed during streaming."
        )
    verify_lfm2_cached_second_order_evidence(request.value_evidence)
    verify_lfm2_cached_second_order_evidence(request.output_evidence)
    verify_lfm2_attention_bridge(request.attention_bridge)
    source_bundle_sha = _canonical_sha256(
        {"value": value_source_sha, "output": output_source_sha}
    )
    evidence_bundle_sha = _canonical_sha256(
        {
            "value": request.value_evidence.evidence_sha256,
            "output": request.output_evidence.evidence_sha256,
            "attention_bridge": request.attention_bridge.bridge_sha256,
        }
    )
    return _finalize(
        site_id=value_name,
        role=value_role,
        semantic_kind=semantic_kind,
        retained=retained,
        generated_per_band=generated_per_band,
        generated_count=generated_count,
        solver_invocations=solver_invocations,
        invariant_statistics_build_count=(invariant_statistics_build_count),
        invariant_target_build_count=invariant_target_build_count,
        candidate_dependent_statistics_build_count=(candidate_statistics_build_count),
        candidate_dependent_target_build_count=(candidate_target_build_count),
        source_bundle_sha256=source_bundle_sha,
        evidence_bundle_sha256=evidence_bundle_sha,
        config=config,
    )


__all__ = [
    "Lfm2PhysicalByteBandReport",
    "Lfm2RetainedCrevfOption",
    "Lfm2StreamingCrevfConfig",
    "Lfm2StreamingCrevfError",
    "Lfm2StreamingCrevfOpsAudit",
    "Lfm2StreamingCrevfSiteBuild",
    "Lfm2StreamingJointAttentionRequest",
    "Lfm2StreamingTensorSiteRequest",
    "build_lfm2_streaming_crevf_site",
    "build_lfm2_streaming_joint_attention_site",
]
