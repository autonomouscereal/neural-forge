"""Authenticated analytic SwiGLU tangent snapshots for LFM2 folding.

The bank is an architecture-derived input to nonlinear empirical
interpolation.  It never executes a model and never differentiates a graph.
For every selected cached activation anchor and every row of an authenticated
child-to-parent normalized-input decoder, it evaluates the exact analytic JVP

``d(silu(g) * u) = silu'(g) * dg * u + silu(g) * du``.

Cached rows are selected only by their canonical row positions.  Activation
magnitudes, spectra, ranks, and sample weights cannot select or reject an
anchor.  The bounded spectrum reported by this module is diagnostic evidence
for a human or a downstream experiment, not a semantic gate.
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

import torch
from torch import Tensor, nn


LFM2_SWIGLU_TANGENT_BANK_SCHEMA = "neural-forge-lfm2-swiglu-tangent-bank-v1"
LFM2_SWIGLU_TANGENT_BANK_MAX_ANCHORS = 4
_ADVISORY_MAX_ROWS = 64
_ADVISORY_MAX_COLUMNS = 128
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class Lfm2SwiGluTangentBankError(ValueError):
    """Tangent evidence is malformed, unauthenticated, or structurally unsafe."""


class Lfm2SwiGluTangentAuthenticationError(Lfm2SwiGluTangentBankError):
    """A tensor differs from its required content digest."""


@dataclass(frozen=True, slots=True)
class Lfm2SwiGluAnchorProvenance:
    """The exact cached source row used as one nonlinear anchor."""

    anchor_slot: int
    source_row_index: int
    source_row_sha256: str
    sample_weight: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "anchor_slot": self.anchor_slot,
            "source_row_index": self.source_row_index,
            "source_row_sha256": self.source_row_sha256,
            "sample_weight": self.sample_weight,
        }


@dataclass(frozen=True, slots=True)
class Lfm2SwiGluTangentProvenance:
    """One bank row's cached anchor and authenticated decoder direction."""

    tangent_row_index: int
    anchor_slot: int
    source_row_index: int
    decoder_row_index: int
    decoder_direction_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "tangent_row_index": self.tangent_row_index,
            "anchor_slot": self.anchor_slot,
            "source_row_index": self.source_row_index,
            "decoder_row_index": self.decoder_row_index,
            "decoder_direction_sha256": self.decoder_direction_sha256,
        }


@dataclass(frozen=True, slots=True)
class Lfm2SwiGluTangentBank:
    """Detached CPU-FP64 tangent rows and their immutable build receipt."""

    site_id: str
    tangent_rows: Tensor
    tangent_row_weights: Tensor
    coordinate_rms_tangent_row_scales: Tensor
    recommended_tangent_row_scales: Tensor
    anchor_indices: tuple[int, ...]
    anchor_provenance: tuple[Lfm2SwiGluAnchorProvenance, ...]
    tangent_provenance: tuple[Lfm2SwiGluTangentProvenance, ...]
    source_hashes: Mapping[str, str]
    tangent_rows_sha256: str
    tangent_row_weights_sha256: str
    coordinate_rms_tangent_row_scales_sha256: str
    coordinate_scaled_tangent_rows_sha256: str
    recommended_tangent_row_scales_sha256: str
    provenance_sha256: str
    bank_sha256: str
    advisory_diagnostics: Mapping[str, Any]
    operation_audit: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready receipt without copying the tangent payload."""

        return {
            "schema": LFM2_SWIGLU_TANGENT_BANK_SCHEMA,
            "site_id": self.site_id,
            "geometry": {
                "anchor_count": len(self.anchor_indices),
                "child_width": int(self.tangent_rows.shape[0])
                // len(self.anchor_indices),
                "parent_intermediate_width": int(self.tangent_rows.shape[1]),
                "tangent_row_count": int(self.tangent_rows.shape[0]),
            },
            "anchor_indices": list(self.anchor_indices),
            "anchor_selection_policy": (
                "fixed_evenly_spaced_canonical_cache_row_positions"
            ),
            "anchor_values_or_numeric_metrics_used_for_selection": False,
            "sample_weights_used_for_anchor_selection": False,
            "direction_policy": (
                "every_authenticated_target_to_parent_decoder_row"
            ),
            "full_parent_coordinate_identity_synthesized": False,
            "source_hashes": dict(self.source_hashes),
            "tangent_rows": {
                "dtype": str(self.tangent_rows.dtype),
                "device": str(self.tangent_rows.device),
                "shape": list(self.tangent_rows.shape),
                "sha256": self.tangent_rows_sha256,
            },
            "tangent_row_weights": {
                "dtype": str(self.tangent_row_weights.dtype),
                "device": str(self.tangent_row_weights.device),
                "shape": list(self.tangent_row_weights.shape),
                "sha256": self.tangent_row_weights_sha256,
            },
            "coordinate_rms_tangent_row_scales": {
                "dtype": str(self.coordinate_rms_tangent_row_scales.dtype),
                "device": str(self.coordinate_rms_tangent_row_scales.device),
                "shape": list(self.coordinate_rms_tangent_row_scales.shape),
                "sha256": self.coordinate_rms_tangent_row_scales_sha256,
                "automatically_applied_to_raw_tangent_rows": False,
            },
            "coordinate_scaled_tangent_rows": {
                "shape": list(self.tangent_rows.shape),
                "sha256": self.coordinate_scaled_tangent_rows_sha256,
                "materialization": (
                    "raw_tangent_rows * repeated_child_input_coordinate_rms"
                ),
                "stored_as_duplicate_payload": False,
            },
            "recommended_tangent_row_scales": {
                "dtype": str(self.recommended_tangent_row_scales.dtype),
                "device": str(self.recommended_tangent_row_scales.device),
                "shape": list(self.recommended_tangent_row_scales.shape),
                "sha256": self.recommended_tangent_row_scales_sha256,
                "decision_role": "advisory_snapshot_balancing_only",
            },
            "anchor_provenance": [
                item.as_dict() for item in self.anchor_provenance
            ],
            "tangent_provenance": [
                item.as_dict() for item in self.tangent_provenance
            ],
            "provenance_sha256": self.provenance_sha256,
            "advisory_diagnostics": _deep_thaw(self.advisory_diagnostics),
            "operation_audit": dict(self.operation_audit),
            "bank_sha256": self.bank_sha256,
        }

    def qdeim_functional_snapshot_kwargs(self, *, block_id: str) -> dict[str, Any]:
        """Return lossless constructor kwargs for the standalone Q-DEIM block.

        This module intentionally does not import the Q-DEIM compiler.  Child
        coordinate RMS is applied directly to the raw direction rows, cached
        sample weights remain unchanged, and global equal-energy balancing is
        represented separately as the compiler's energy ``block_weight``.
        """

        if not isinstance(block_id, str) or not block_id.strip():
            raise Lfm2SwiGluTangentBankError(
                "Q-DEIM functional snapshot block_id must be nonempty."
            )
        recommendation = self.advisory_diagnostics[
            "snapshot_balance_recommendation"
        ]
        global_amplitude_scale = float(
            recommendation["recommended_tangent_global_scale"]
        )
        block_weight = global_amplitude_scale * global_amplitude_scale
        functional_rows = (
            self.tangent_rows
            * self.coordinate_rms_tangent_row_scales.unsqueeze(1)
        ).contiguous()
        if (
            fingerprint_lfm2_swiglu_tangent_tensor(functional_rows)
            != self.coordinate_scaled_tangent_rows_sha256
        ):
            raise Lfm2SwiGluTangentAuthenticationError(
                "Coordinate-scaled tangent rows changed after bank sealing."
            )
        return {
            "block_id": block_id.strip(),
            "snapshot_kind": "analytic_child_lifted_parent_swiglu_tangent",
            "functional_rows": functional_rows,
            "generator_receipt_sha256": self.bank_sha256,
            "block_weight": block_weight,
            "row_sample_weights": tuple(
                float(value) for value in self.tangent_row_weights.tolist()
            ),
        }


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_canonical(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    if isinstance(value, list):
        return [_deep_thaw(item) for item in value]
    return value


def _tensor_raw_bytes(value: Tensor) -> bytes:
    cpu = value.detach().to(device="cpu").contiguous()
    return cpu.view(torch.uint8).numpy().tobytes(order="C")


def fingerprint_lfm2_swiglu_tangent_tensor(value: Tensor) -> str:
    """Return a stable dtype/shape/raw-bytes SHA-256 tensor fingerprint."""

    if not isinstance(value, Tensor):
        raise Lfm2SwiGluTangentBankError("value must be a Tensor.")
    if value.layout != torch.strided or value.device.type == "meta":
        raise Lfm2SwiGluTangentBankError(
            "value must be a materialized strided Tensor."
        )
    header = _canonical_bytes(
        {
            "dtype": str(value.dtype),
            "shape": tuple(int(part) for part in value.shape),
        }
    )
    return _sha256_bytes(header + b"\0" + _tensor_raw_bytes(value))


def _require_digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2SwiGluTangentAuthenticationError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _require_plain_finite_tensor(value: Any, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise Lfm2SwiGluTangentBankError(f"{name} must be a Tensor.")
    if isinstance(value, nn.Parameter):
        raise Lfm2SwiGluTangentBankError(f"{name} cannot be an nn.Parameter.")
    if value.requires_grad:
        raise Lfm2SwiGluTangentBankError(
            f"{name} cannot require gradients."
        )
    if (
        value.layout != torch.strided
        or value.device.type == "meta"
        or not value.dtype.is_floating_point
    ):
        raise Lfm2SwiGluTangentBankError(
            f"{name} must be a materialized strided floating-point Tensor."
        )
    if value.numel() == 0 or not bool(torch.isfinite(value).all()):
        raise Lfm2SwiGluTangentBankError(f"{name} must be nonempty and finite.")
    return value


def _authenticate_tensor(value: Tensor, expected: Any, name: str) -> str:
    required = _require_digest(expected, f"{name}_sha256")
    observed = fingerprint_lfm2_swiglu_tangent_tensor(value)
    if observed != required:
        raise Lfm2SwiGluTangentAuthenticationError(
            f"{name} differs from its authenticated SHA-256 digest."
        )
    return observed


def _positive_small_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2SwiGluTangentBankError(f"{name} must be a positive integer.")
    if value > LFM2_SWIGLU_TANGENT_BANK_MAX_ANCHORS:
        raise Lfm2SwiGluTangentBankError(
            f"{name} cannot exceed the structural bound "
            f"{LFM2_SWIGLU_TANGENT_BANK_MAX_ANCHORS}."
        )
    return value


def _structural_indices(length: int, count: int) -> tuple[int, ...]:
    count = min(length, count)
    if count == 1:
        return (length // 2,)
    return tuple(
        (slot * (length - 1)) // (count - 1) for slot in range(count)
    )


def _source_row_sha256(gate: Tensor, up: Tensor, post: Tensor) -> str:
    return _sha256_canonical(
        {
            "gate_sha256": fingerprint_lfm2_swiglu_tangent_tensor(gate),
            "up_sha256": fingerprint_lfm2_swiglu_tangent_tensor(up),
            "post_sha256": fingerprint_lfm2_swiglu_tangent_tensor(post),
        }
    )


def _advisory_spectrum(tangent_rows: Tensor) -> dict[str, Any]:
    """Compute a bounded deterministic structural-submatrix diagnostic."""

    row_indices = _structural_indices(
        int(tangent_rows.shape[0]),
        min(int(tangent_rows.shape[0]), _ADVISORY_MAX_ROWS),
    )
    column_indices = _structural_indices(
        int(tangent_rows.shape[1]),
        min(int(tangent_rows.shape[1]), _ADVISORY_MAX_COLUMNS),
    )
    rows = torch.tensor(row_indices, dtype=torch.long)
    columns = torch.tensor(column_indices, dtype=torch.long)
    sketch = tangent_rows.index_select(0, rows).index_select(1, columns)
    diagnostic: dict[str, Any] = {
        "kind": "deterministic_evenly_spaced_structural_submatrix_svd",
        "decision_role": "advisory_only",
        "used_for_anchor_selection": False,
        "used_for_semantic_gate": False,
        "used_for_auto_veto": False,
        "covers_full_tangent_bank": (
            len(row_indices) == int(tangent_rows.shape[0])
            and len(column_indices) == int(tangent_rows.shape[1])
        ),
        "sketch_row_indices": list(row_indices),
        "sketch_column_indices": list(column_indices),
        "sketch_shape": list(sketch.shape),
    }
    try:
        singular_values = torch.linalg.svdvals(sketch)
        if not bool(torch.isfinite(singular_values).all()):
            raise RuntimeError("The advisory singular spectrum is nonfinite.")
        largest = float(singular_values[0]) if singular_values.numel() else 0.0
        tolerance = (
            float(torch.finfo(torch.float64).eps)
            * max(int(sketch.shape[0]), int(sketch.shape[1]))
            * largest
        )
        rank = int((singular_values > tolerance).sum().item())
        diagnostic.update(
            {
                "status": "available",
                "rank_estimate": rank,
                "rank_tolerance": tolerance,
                "singular_values": [
                    float(value) for value in singular_values.tolist()
                ],
            }
        )
    except RuntimeError as exc:
        # A diagnostic backend failure must not veto an otherwise valid bank.
        diagnostic.update(
            {
                "status": "unavailable",
                "failure_type": type(exc).__name__,
                "rank_estimate": None,
                "rank_tolerance": None,
                "singular_values": [],
            }
        )
    return diagnostic


def _advisory_diagnostics(
    *,
    tangent_rows: Tensor,
    cached_post_rows: Tensor,
    sample_weights: Tensor,
    selected_sample_weights: Tensor,
    child_input_coordinate_rms: Tensor,
) -> tuple[Mapping[str, Any], Tensor, Tensor]:
    """Describe amplitude geometry and recommend, but never impose, balancing."""

    anchor_count = int(selected_sample_weights.shape[0])
    child_width = int(child_input_coordinate_rms.shape[0])
    intermediate_width = int(tangent_rows.shape[1])
    blocks = tangent_rows.reshape(anchor_count, child_width, intermediate_width)
    per_anchor = torch.linalg.vector_norm(blocks.reshape(anchor_count, -1), dim=1)
    per_direction = torch.linalg.vector_norm(
        blocks.permute(1, 0, 2).reshape(child_width, -1), dim=1
    )

    weighted_post_norm = torch.linalg.vector_norm(
        cached_post_rows * torch.sqrt(sample_weights).unsqueeze(1)
    )
    weighted_tangent_norm = torch.linalg.vector_norm(
        blocks
        * torch.sqrt(selected_sample_weights).reshape(anchor_count, 1, 1)
        * child_input_coordinate_rms.reshape(1, child_width, 1)
    )
    post_norm = float(weighted_post_norm)
    tangent_norm = float(weighted_tangent_norm)
    candidate_scale = (
        post_norm / tangent_norm
        if (
            math.isfinite(post_norm)
            and math.isfinite(tangent_norm)
            and post_norm > 0.0
            and tangent_norm > 0.0
        )
        else None
    )
    if (
        candidate_scale is not None
        and math.isfinite(candidate_scale)
        and candidate_scale > 0.0
    ):
        global_scale = candidate_scale
        balance_status = "available_exact_weighted_frobenius_equalization"
    else:
        # Exact zero energy makes a multiplicative equalization undefined.  A
        # unit global scale preserves the raw bank and records that no
        # recommendation was available; it does not reject the bank.
        global_scale = 1.0
        balance_status = "unavailable_zero_or_nonfinite_advisory_block_norm"
    coordinate_row_scales = (
        child_input_coordinate_rms.repeat(anchor_count).contiguous().clone()
    )
    row_scales = (coordinate_row_scales * global_scale).contiguous().clone()
    diagnostic: dict[str, Any] = {
        "decision_role": "advisory_only",
        "used_for_anchor_selection": False,
        "used_for_semantic_gate": False,
        "used_for_auto_veto": False,
        "raw_tangent_frobenius_norm_by_anchor": [
            value if math.isfinite(value) else None
            for value in (float(item) for item in per_anchor.tolist())
        ],
        "raw_tangent_frobenius_norm_by_decoder_direction": [
            value if math.isfinite(value) else None
            for value in (float(item) for item in per_direction.tolist())
        ],
        "snapshot_balance_recommendation": {
            "status": balance_status,
            "policy": (
                "equal_total_weighted_frobenius_norm_of_empirical_post_and_"
                "coordinate_rms_scaled_tangent_blocks"
            ),
            "empirical_post_weighted_frobenius_norm": (
                post_norm if math.isfinite(post_norm) else None
            ),
            "coordinate_rms_scaled_tangent_weighted_frobenius_norm_before_"
            "balance": tangent_norm if math.isfinite(tangent_norm) else None,
            "recommended_tangent_global_scale": global_scale,
            "recommended_row_scale_formula": (
                "child_input_coordinate_rms[decoder_row] * global_scale"
            ),
            "sample_weights_enter_as_row_weights_not_row_scales": True,
            "recommendation_is_not_automatically_applied": True,
            "recommendation_is_not_a_semantic_gate": True,
        },
        "spectrum": _advisory_spectrum(tangent_rows),
    }
    diagnostic["diagnostic_sha256"] = _sha256_canonical(diagnostic)
    return MappingProxyType(diagnostic), coordinate_row_scales, row_scales


def _build_provenance(
    *,
    anchor_indices: Sequence[int],
    cached_gate: Tensor,
    cached_up: Tensor,
    cached_post: Tensor,
    decoder: Tensor,
    sample_weights: Tensor,
) -> tuple[
    tuple[Lfm2SwiGluAnchorProvenance, ...],
    tuple[Lfm2SwiGluTangentProvenance, ...],
]:
    anchor_records: list[Lfm2SwiGluAnchorProvenance] = []
    tangent_records: list[Lfm2SwiGluTangentProvenance] = []
    direction_hashes = tuple(
        fingerprint_lfm2_swiglu_tangent_tensor(decoder[index])
        for index in range(int(decoder.shape[0]))
    )
    child_width = int(decoder.shape[0])
    for anchor_slot, source_row_index in enumerate(anchor_indices):
        anchor_records.append(
            Lfm2SwiGluAnchorProvenance(
                anchor_slot=anchor_slot,
                source_row_index=source_row_index,
                source_row_sha256=_source_row_sha256(
                    cached_gate[source_row_index],
                    cached_up[source_row_index],
                    cached_post[source_row_index],
                ),
                sample_weight=float(sample_weights[source_row_index]),
            )
        )
        for decoder_row_index, direction_sha256 in enumerate(direction_hashes):
            tangent_records.append(
                Lfm2SwiGluTangentProvenance(
                    tangent_row_index=anchor_slot * child_width + decoder_row_index,
                    anchor_slot=anchor_slot,
                    source_row_index=source_row_index,
                    decoder_row_index=decoder_row_index,
                    decoder_direction_sha256=direction_sha256,
                )
            )
    return tuple(anchor_records), tuple(tangent_records)


def build_lfm2_swiglu_tangent_bank(
    *,
    site_id: str,
    parent_w1: Tensor,
    parent_w1_sha256: str,
    parent_w3: Tensor,
    parent_w3_sha256: str,
    cached_gate_rows: Tensor,
    cached_gate_rows_sha256: str,
    cached_up_rows: Tensor,
    cached_up_rows_sha256: str,
    cached_post_rows: Tensor,
    cached_post_rows_sha256: str,
    target_to_parent_decoder: Tensor,
    target_to_parent_decoder_sha256: str,
    sample_weights: Tensor | None = None,
    sample_weights_sha256: str | None = None,
    child_input_coordinate_rms: Tensor | None = None,
    child_input_coordinate_rms_sha256: str | None = None,
    anchor_count: int = 3,
) -> Lfm2SwiGluTangentBank:
    """Build exact analytic SwiGLU JVP rows from sealed parent evidence.

    ``target_to_parent_decoder`` has shape ``[child_width, parent_hidden]``.
    Its rows, and only its rows, are the tangent directions.  This function
    never manufactures a full parent-coordinate identity basis.
    """

    if not isinstance(site_id, str) or not site_id.strip():
        raise Lfm2SwiGluTangentBankError("site_id must be a nonempty string.")
    site_id = site_id.strip()
    requested_anchor_count = _positive_small_int(anchor_count, "anchor_count")

    named_tensors = {
        "parent_w1": _require_plain_finite_tensor(parent_w1, "parent_w1"),
        "parent_w3": _require_plain_finite_tensor(parent_w3, "parent_w3"),
        "cached_gate_rows": _require_plain_finite_tensor(
            cached_gate_rows, "cached_gate_rows"
        ),
        "cached_up_rows": _require_plain_finite_tensor(
            cached_up_rows, "cached_up_rows"
        ),
        "cached_post_rows": _require_plain_finite_tensor(
            cached_post_rows, "cached_post_rows"
        ),
        "target_to_parent_decoder": _require_plain_finite_tensor(
            target_to_parent_decoder, "target_to_parent_decoder"
        ),
    }
    expected_hashes = {
        "parent_w1": parent_w1_sha256,
        "parent_w3": parent_w3_sha256,
        "cached_gate_rows": cached_gate_rows_sha256,
        "cached_up_rows": cached_up_rows_sha256,
        "cached_post_rows": cached_post_rows_sha256,
        "target_to_parent_decoder": target_to_parent_decoder_sha256,
    }
    source_hashes_before = {
        name: _authenticate_tensor(value, expected_hashes[name], name)
        for name, value in named_tensors.items()
    }

    if sample_weights is None:
        if sample_weights_sha256 is not None:
            raise Lfm2SwiGluTangentAuthenticationError(
                "sample_weights_sha256 must be omitted when sample_weights is omitted."
            )
        weights_source_sha256 = "none"
    else:
        sample_weights = _require_plain_finite_tensor(
            sample_weights, "sample_weights"
        )
        if sample_weights_sha256 is None:
            raise Lfm2SwiGluTangentAuthenticationError(
                "sample_weights_sha256 is required with sample_weights."
            )
        weights_source_sha256 = _authenticate_tensor(
            sample_weights, sample_weights_sha256, "sample_weights"
        )
        named_tensors["sample_weights"] = sample_weights
        source_hashes_before["sample_weights"] = weights_source_sha256

    if child_input_coordinate_rms is None:
        if child_input_coordinate_rms_sha256 is not None:
            raise Lfm2SwiGluTangentAuthenticationError(
                "child_input_coordinate_rms_sha256 must be omitted when "
                "child_input_coordinate_rms is omitted."
            )
        coordinate_rms_source_sha256 = "none"
    else:
        child_input_coordinate_rms = _require_plain_finite_tensor(
            child_input_coordinate_rms, "child_input_coordinate_rms"
        )
        if child_input_coordinate_rms_sha256 is None:
            raise Lfm2SwiGluTangentAuthenticationError(
                "child_input_coordinate_rms_sha256 is required with "
                "child_input_coordinate_rms."
            )
        coordinate_rms_source_sha256 = _authenticate_tensor(
            child_input_coordinate_rms,
            child_input_coordinate_rms_sha256,
            "child_input_coordinate_rms",
        )
        named_tensors["child_input_coordinate_rms"] = child_input_coordinate_rms
        source_hashes_before["child_input_coordinate_rms"] = (
            coordinate_rms_source_sha256
        )

    for name in (
        "parent_w1",
        "parent_w3",
        "cached_gate_rows",
        "cached_up_rows",
        "cached_post_rows",
        "target_to_parent_decoder",
    ):
        if named_tensors[name].ndim != 2:
            raise Lfm2SwiGluTangentBankError(f"{name} must be rank two.")

    w1_shape = tuple(int(part) for part in parent_w1.shape)
    if tuple(int(part) for part in parent_w3.shape) != w1_shape:
        raise Lfm2SwiGluTangentBankError(
            "parent_w1 and parent_w3 must share [intermediate, hidden] geometry."
        )
    intermediate_width, parent_hidden = w1_shape
    cache_shape = tuple(int(part) for part in cached_gate_rows.shape)
    if (
        tuple(int(part) for part in cached_up_rows.shape) != cache_shape
        or tuple(int(part) for part in cached_post_rows.shape) != cache_shape
    ):
        raise Lfm2SwiGluTangentBankError(
            "cached gate, up, and post rows must share exact geometry."
        )
    sample_count, cached_intermediate = cache_shape
    if cached_intermediate != intermediate_width:
        raise Lfm2SwiGluTangentBankError(
            "cached rows do not match the authenticated parent intermediate width."
        )
    decoder_shape = tuple(int(part) for part in target_to_parent_decoder.shape)
    child_width, decoder_parent_hidden = decoder_shape
    if decoder_parent_hidden != parent_hidden:
        raise Lfm2SwiGluTangentBankError(
            "target_to_parent_decoder must have shape "
            "[child_width, parent_hidden]."
        )

    if sample_weights is None:
        weights_cpu = torch.ones(sample_count, dtype=torch.float64)
    else:
        if sample_weights.ndim != 1 or int(sample_weights.shape[0]) != sample_count:
            raise Lfm2SwiGluTangentBankError(
                "sample_weights must have shape [cached_row_count]."
            )
        weights_cpu = (
            sample_weights.detach()
            .to(device="cpu", dtype=torch.float64)
            .contiguous()
            .clone()
        )
        if not bool((weights_cpu > 0.0).all()):
            raise Lfm2SwiGluTangentBankError(
                "sample_weights must be strictly positive."
            )

    if child_input_coordinate_rms is None:
        coordinate_rms_cpu = torch.ones(child_width, dtype=torch.float64)
    else:
        if (
            child_input_coordinate_rms.ndim != 1
            or int(child_input_coordinate_rms.shape[0]) != child_width
        ):
            raise Lfm2SwiGluTangentBankError(
                "child_input_coordinate_rms must have shape [child_width]."
            )
        coordinate_rms_cpu = (
            child_input_coordinate_rms.detach()
            .to(device="cpu", dtype=torch.float64)
            .contiguous()
            .clone()
        )
        if not bool((coordinate_rms_cpu > 0.0).all()):
            raise Lfm2SwiGluTangentBankError(
                "child_input_coordinate_rms must be strictly positive."
            )

    # Clone every source before arithmetic.  The original tensors are touched
    # again only for the post-build content authentication below.
    w1_cpu = parent_w1.detach().to(device="cpu", dtype=torch.float64).clone()
    w3_cpu = parent_w3.detach().to(device="cpu", dtype=torch.float64).clone()
    gate_cpu = (
        cached_gate_rows.detach().to(device="cpu", dtype=torch.float64).clone()
    )
    up_cpu = cached_up_rows.detach().to(device="cpu", dtype=torch.float64).clone()
    post_cpu = cached_post_rows.detach().to(device="cpu", dtype=torch.float64).clone()
    decoder_cpu = (
        target_to_parent_decoder.detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
        .clone()
    )
    anchor_indices = _structural_indices(sample_count, requested_anchor_count)

    with torch.inference_mode():
        # These projections depend only on authenticated decoder rows.  There
        # is intentionally no parent-coordinate identity fallback.
        direction_gate = decoder_cpu @ w1_cpu.transpose(0, 1)
        direction_up = decoder_cpu @ w3_cpu.transpose(0, 1)
        blocks: list[Tensor] = []
        for source_row_index in anchor_indices:
            gate = gate_cpu[source_row_index]
            up = up_cpu[source_row_index]
            sigmoid_gate = torch.sigmoid(gate)
            silu_gate = gate * sigmoid_gate
            silu_prime = sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
            blocks.append(
                direction_gate * (silu_prime * up).unsqueeze(0)
                + direction_up * silu_gate.unsqueeze(0)
            )
        tangent_rows = torch.cat(blocks, dim=0).contiguous().clone()
        tangent_row_weights = (
            weights_cpu[torch.tensor(anchor_indices, dtype=torch.long)]
            .repeat_interleave(child_width)
            .contiguous()
            .clone()
        )

    if not bool(torch.isfinite(tangent_rows).all()):
        raise Lfm2SwiGluTangentBankError(
            "Analytic SwiGLU tangent rows overflowed or became nonfinite."
        )
    if tangent_rows.device.type != "cpu" or tangent_rows.dtype != torch.float64:
        raise AssertionError("Internal tangent representation must be CPU FP64.")
    tangent_rows.requires_grad_(False)
    tangent_row_weights.requires_grad_(False)

    anchor_provenance, tangent_provenance = _build_provenance(
        anchor_indices=anchor_indices,
        cached_gate=gate_cpu,
        cached_up=up_cpu,
        cached_post=post_cpu,
        decoder=decoder_cpu,
        sample_weights=weights_cpu,
    )
    provenance_payload = {
        "anchors": [item.as_dict() for item in anchor_provenance],
        "tangents": [item.as_dict() for item in tangent_provenance],
    }
    provenance_sha256 = _sha256_canonical(provenance_payload)
    tangent_rows_sha256 = fingerprint_lfm2_swiglu_tangent_tensor(tangent_rows)
    tangent_row_weights_sha256 = fingerprint_lfm2_swiglu_tangent_tensor(
        tangent_row_weights
    )

    source_hashes_after = {
        name: fingerprint_lfm2_swiglu_tangent_tensor(value)
        for name, value in named_tensors.items()
    }
    if source_hashes_after != source_hashes_before:
        raise Lfm2SwiGluTangentAuthenticationError(
            "A source tensor changed during tangent-bank construction."
        )
    source_hashes = dict(source_hashes_before)
    source_hashes["sample_weights"] = weights_source_sha256
    source_hashes["child_input_coordinate_rms"] = coordinate_rms_source_sha256

    selected_weights = weights_cpu[
        torch.tensor(anchor_indices, dtype=torch.long)
    ].contiguous()
    (
        advisory_diagnostics,
        coordinate_rms_tangent_row_scales,
        recommended_tangent_row_scales,
    ) = _advisory_diagnostics(
        tangent_rows=tangent_rows,
        cached_post_rows=post_cpu,
        sample_weights=weights_cpu,
        selected_sample_weights=selected_weights,
        child_input_coordinate_rms=coordinate_rms_cpu,
    )
    recommended_tangent_row_scales.requires_grad_(False)
    coordinate_rms_tangent_row_scales.requires_grad_(False)
    coordinate_rms_tangent_row_scales_sha256 = (
        fingerprint_lfm2_swiglu_tangent_tensor(
            coordinate_rms_tangent_row_scales
        )
    )
    coordinate_scaled_tangent_rows_sha256 = (
        fingerprint_lfm2_swiglu_tangent_tensor(
            (
                tangent_rows
                * coordinate_rms_tangent_row_scales.unsqueeze(1)
            ).contiguous()
        )
    )
    recommended_tangent_row_scales_sha256 = (
        fingerprint_lfm2_swiglu_tangent_tensor(recommended_tangent_row_scales)
    )
    operation_audit = MappingProxyType(
        {
            "parent_forward_calls": 0,
            "model_forward_calls": 0,
            "generation_calls": 0,
            "backward_calls": 0,
            "autograd_grad_calls": 0,
            "optimizer_constructions": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
            "source_tensor_mutations": 0,
            "source_hashes_equal_before_and_after": True,
            "analytic_jvp_rows": int(tangent_rows.shape[0]),
            "authenticated_decoder_direction_rows": child_width,
            "synthesized_parent_identity_direction_rows": 0,
            "detached_cpu_float64_output": True,
            "numeric_diagnostics_used_for_semantic_decision": False,
            "numeric_diagnostics_used_for_auto_veto": False,
            "recommended_snapshot_scales_automatically_applied": False,
        }
    )
    bank_payload = {
        "schema": LFM2_SWIGLU_TANGENT_BANK_SCHEMA,
        "site_id": site_id,
        "source_hashes": source_hashes,
        "anchor_indices": anchor_indices,
        "anchor_selection_policy": (
            "fixed_evenly_spaced_canonical_cache_row_positions"
        ),
        "direction_policy": "every_authenticated_target_to_parent_decoder_row",
        "geometry": {
            "sample_count": sample_count,
            "parent_hidden": parent_hidden,
            "parent_intermediate": intermediate_width,
            "child_width": child_width,
            "tangent_row_count": int(tangent_rows.shape[0]),
        },
        "tangent_rows_sha256": tangent_rows_sha256,
        "tangent_row_weights_sha256": tangent_row_weights_sha256,
        "recommended_tangent_row_scales_sha256": (
            recommended_tangent_row_scales_sha256
        ),
        "coordinate_rms_tangent_row_scales_sha256": (
            coordinate_rms_tangent_row_scales_sha256
        ),
        "coordinate_scaled_tangent_rows_sha256": (
            coordinate_scaled_tangent_rows_sha256
        ),
        "provenance_sha256": provenance_sha256,
    }
    bank_sha256 = _sha256_canonical(bank_payload)
    return Lfm2SwiGluTangentBank(
        site_id=site_id,
        tangent_rows=tangent_rows,
        tangent_row_weights=tangent_row_weights,
        coordinate_rms_tangent_row_scales=coordinate_rms_tangent_row_scales,
        recommended_tangent_row_scales=recommended_tangent_row_scales,
        anchor_indices=anchor_indices,
        anchor_provenance=anchor_provenance,
        tangent_provenance=tangent_provenance,
        source_hashes=MappingProxyType(source_hashes),
        tangent_rows_sha256=tangent_rows_sha256,
        tangent_row_weights_sha256=tangent_row_weights_sha256,
        coordinate_rms_tangent_row_scales_sha256=(
            coordinate_rms_tangent_row_scales_sha256
        ),
        coordinate_scaled_tangent_rows_sha256=(
            coordinate_scaled_tangent_rows_sha256
        ),
        recommended_tangent_row_scales_sha256=(
            recommended_tangent_row_scales_sha256
        ),
        provenance_sha256=provenance_sha256,
        bank_sha256=bank_sha256,
        advisory_diagnostics=advisory_diagnostics,
        operation_audit=operation_audit,
    )


__all__ = [
    "LFM2_SWIGLU_TANGENT_BANK_MAX_ANCHORS",
    "LFM2_SWIGLU_TANGENT_BANK_SCHEMA",
    "Lfm2SwiGluAnchorProvenance",
    "Lfm2SwiGluTangentAuthenticationError",
    "Lfm2SwiGluTangentBank",
    "Lfm2SwiGluTangentBankError",
    "Lfm2SwiGluTangentProvenance",
    "build_lfm2_swiglu_tangent_bank",
    "fingerprint_lfm2_swiglu_tangent_tensor",
]
