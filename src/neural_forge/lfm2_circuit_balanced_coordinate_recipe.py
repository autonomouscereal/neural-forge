"""Authenticated circuit-balanced evidence for the LFM2 coordinate recipe.

This adapter deliberately does not install a second residual coordinate map.
It turns the already-proven orthonormal circuit-balanced basis into one
source-space orthogonal projector consumed by analytic coordinate completion.
The semantic-foundation token table and producer encoder remain the global
interface gauge.  In particular, no non-orthogonal balanced realization is
silently transported through RMSNorm.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch

from neural_forge.dynamic_circuit_ir import ModelCircuitGraph, canonical_sha256
from neural_forge.hf_lfm2_balanced_circuit_basis import (
    Lfm2CircuitBalancedBasisResult,
    build_lfm2_circuit_balanced_basis,
)
from neural_forge.hf_lfm2_bridge_only_control import (
    _tensor_sha256 as _causal_table_sha256,
)
from neural_forge.hf_lfm2_interface_frontier import (
    LFM2_UNSUPPORTED_LOGIT_POLICY_ADAPTIVE_STRONGEST_SURVIVING_BACKGROUND_SEAL,
)
from neural_forge.hf_tied_causal_interface_fold import (
    TiedCausalInterfaceFold,
    _tensor_sha256 as _interface_tensor_sha256,
)
from neural_forge.hf_tokenizer_bridge import StrictTokenizerBridgeArtifact


LFM2_CIRCUIT_BALANCED_COORDINATE_RECIPE = (
    "circuit-balanced-vocabulary-margin-bridge-procrustes-isometric"
)
LFM2_CIRCUIT_BALANCED_COORDINATE_EVIDENCE_SCHEMA = (
    "neural-forge-lfm2-circuit-balanced-coordinate-evidence-v1"
)

_HEX_DIGITS = frozenset("0123456789abcdef")
_REQUIRED_STRUCTURAL_ROLES = frozenset(
    {
        "lexical_interface",
        "residual_normalization",
        "swiglu_input_sensitivity",
        "swiglu_output_span",
        "attention_qkv_input_sensitivity",
        "attention_output_span",
        "short_conv_input_sensitivity",
        "short_conv_output_span",
        "short_conv_state_channel",
    }
)


class Lfm2CircuitBalancedCoordinateEvidenceError(ValueError):
    """Circuit-balanced coordinate evidence is incomplete or unauthenticated."""


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise Lfm2CircuitBalancedCoordinateEvidenceError(
            f"{label} must be one lowercase SHA-256 digest."
        )
    return value


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    if tensor.is_conj():
        tensor = tensor.resolve_conj()
    if tensor.is_neg():
        tensor = tensor.resolve_neg()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(
        json.dumps(
            [int(size) for size in tensor.shape],
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _causal_anchor_tensor_sha256(value: torch.Tensor) -> str:
    """Hash tensors in the authenticated causal-anchor report domain.

    ``TiedCausalInterfaceFold`` is also the carrier for analytically repaired
    causal-anchor interfaces.  Those reports predate the tied-interface hash
    format and intentionally bind tensors with a canonical descriptor followed
    by the raw bytes.  A coordinate recipe must authenticate the report in the
    domain declared by its schema instead of assuming every valid carrier is a
    base tied-interface report.
    """

    tensor = value.detach().to(device="cpu").contiguous()
    descriptor = {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": [int(size) for size in tensor.shape],
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
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


_BASE_INTERFACE_REPORT_SCHEMAS = frozenset(
    {
        "neural-forge-tied-causal-interface-fold-v1",
        "neural-forge-tied-causal-interface-self-dual-gauge-v1",
    }
)
_CAUSAL_ANCHOR_INTERFACE_REPORT_SCHEMAS = frozenset(
    {"neural-forge-tied-causal-anchor-residual-v1"}
)


def _interface_report_tensor_bindings(
    interface: TiedCausalInterfaceFold,
    report: Mapping[str, Any],
) -> tuple[str, str]:
    """Return the tensor seals required by the report's authenticated schema.

    Unknown extension schemas remain usable when they unambiguously bind both
    tensors in one of the two established domains.  Known schemas are strict:
    they cannot switch hash domains while retaining a freshly computed JSON
    seal.
    """

    base_bindings = (
        _interface_tensor_sha256(interface.producer_encoder),
        _interface_tensor_sha256(interface.target_weight),
    )
    anchor_bindings = (
        _causal_anchor_tensor_sha256(interface.producer_encoder),
        _causal_anchor_tensor_sha256(interface.target_weight),
    )
    schema = report.get("schema")
    if schema in _BASE_INTERFACE_REPORT_SCHEMAS:
        return base_bindings
    if schema in _CAUSAL_ANCHOR_INTERFACE_REPORT_SCHEMAS:
        return anchor_bindings

    supplied = (
        report.get("producer_encoder_sha256"),
        report.get("target_weight_sha256"),
    )
    matches = tuple(
        bindings
        for bindings in (base_bindings, anchor_bindings)
        if supplied == bindings
    )
    if len(matches) != 1:
        raise Lfm2CircuitBalancedCoordinateEvidenceError(
            "The semantic-foundation interface report has an unknown tensor "
            "binding domain."
        )
    return matches[0]


def _matrix(value: Any, label: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or not torch.is_floating_point(value)
        or torch.is_complex(value)
        or value.ndim != 2
        or any(int(size) <= 0 for size in value.shape)
    ):
        raise Lfm2CircuitBalancedCoordinateEvidenceError(
            f"{label} must be one materialized dense floating matrix."
        )
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous().clone()
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2CircuitBalancedCoordinateEvidenceError(
            f"{label} contains non-finite values."
        )
    return result


def _unit_interval(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or not 0.0 < float(value) < 1.0
    ):
        raise Lfm2CircuitBalancedCoordinateEvidenceError(
            f"{label} must be finite and strictly between zero and one."
        )
    return float(value)


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2CircuitBalancedCoordinateEvidenceError(
            f"{label} must be a positive integer."
        )
    return value


def _sealed_interface_report(interface: TiedCausalInterfaceFold) -> dict[str, Any]:
    if not isinstance(interface, TiedCausalInterfaceFold):
        raise Lfm2CircuitBalancedCoordinateEvidenceError(
            "foundation_interface must be a TiedCausalInterfaceFold."
        )
    report = dict(interface.report)
    seal = _digest(report.pop("report_sha256", None), "interface report SHA-256")
    if canonical_sha256(report) != seal:
        raise Lfm2CircuitBalancedCoordinateEvidenceError(
            "The semantic-foundation interface report seal is invalid."
        )
    producer_sha256, target_sha256 = _interface_report_tensor_bindings(
        interface,
        report,
    )
    if (
        report.get("producer_encoder_sha256") != producer_sha256
        or report.get("target_weight_sha256") != target_sha256
    ):
        raise Lfm2CircuitBalancedCoordinateEvidenceError(
            "The semantic-foundation interface report does not bind its tensors."
        )
    return {**report, "report_sha256": seal}


@dataclass(frozen=True, slots=True)
class Lfm2CircuitBalancedCoordinateEvidence:
    """One orthogonal source projector and its complete authenticated lineage."""

    balanced_basis: Lfm2CircuitBalancedBasisResult
    source_projector: torch.Tensor
    completion_weight: float
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.balanced_basis, Lfm2CircuitBalancedBasisResult):
            raise Lfm2CircuitBalancedCoordinateEvidenceError(
                "balanced_basis must be an Lfm2CircuitBalancedBasisResult."
            )
        projector = _matrix(self.source_projector, "source_projector")
        weight = _unit_interval(self.completion_weight, "completion_weight")
        source_width = self.balanced_basis.source_width
        target_width = self.balanced_basis.target_width
        if tuple(projector.shape) != (source_width, source_width):
            raise Lfm2CircuitBalancedCoordinateEvidenceError(
                "source_projector must be source-width square."
            )
        basis = _matrix(self.balanced_basis.basis, "balanced basis")
        identity = torch.eye(target_width, dtype=torch.float64)
        expected_projector = (basis @ basis.T).contiguous()
        orthonormal_error = float((basis.T @ basis - identity).abs().max().item())
        symmetry_error = float((projector - projector.T).abs().max().item())
        idempotence_error = float((projector @ projector - projector).abs().max().item())
        if (
            orthonormal_error > 1e-10
            or symmetry_error > 1e-10
            or idempotence_error > 1e-10
            or not torch.equal(projector, expected_projector)
        ):
            raise Lfm2CircuitBalancedCoordinateEvidenceError(
                "The balanced coordinate projector is not the authenticated "
                "orthogonal basis projector."
            )
        if not isinstance(self.receipt, Mapping):
            raise Lfm2CircuitBalancedCoordinateEvidenceError(
                "receipt must be a mapping."
            )
        receipt = dict(self.receipt)
        if receipt.get("schema") != LFM2_CIRCUIT_BALANCED_COORDINATE_EVIDENCE_SCHEMA:
            raise Lfm2CircuitBalancedCoordinateEvidenceError(
                "Circuit-balanced coordinate evidence schema is invalid."
            )
        seal = _digest(
            receipt.pop("result_sha256", None),
            "circuit-balanced coordinate evidence result SHA-256",
        )
        if canonical_sha256(receipt) != seal:
            raise Lfm2CircuitBalancedCoordinateEvidenceError(
                "Circuit-balanced coordinate evidence seal is invalid."
            )
        rmsnorm = receipt.get("rmsnorm_isometry_contract")
        if (
            receipt.get("recipe") != LFM2_CIRCUIT_BALANCED_COORDINATE_RECIPE
            or receipt.get("source_width") != source_width
            or receipt.get("target_width") != target_width
            or receipt.get("basis_sha256") != self.balanced_basis.map_sha256
            or receipt.get("source_projector_sha256") != _tensor_sha256(projector)
            or receipt.get("completion_weight") != weight
            or not isinstance(rmsnorm, Mapping)
            or rmsnorm.get("orthonormal_basis_isometry") is not True
            or rmsnorm.get("orthogonal_projector_only_crosses_norm_boundary") is not True
            or rmsnorm.get("nonorthogonal_balanced_factors_cross_rmsnorm") is not False
            or rmsnorm.get(
                "nonorthogonal_factors_absorbed_into_adjacent_linear_operators"
            )
            is not True
        ):
            raise Lfm2CircuitBalancedCoordinateEvidenceError(
                "Circuit-balanced coordinate evidence contradicts its basis contract."
            )
        object.__setattr__(self, "source_projector", projector)
        object.__setattr__(self, "completion_weight", weight)
        object.__setattr__(
            self,
            "receipt",
            MappingProxyType({**receipt, "result_sha256": seal}),
        )

    @property
    def source_width(self) -> int:
        return self.balanced_basis.source_width

    @property
    def target_width(self) -> int:
        return self.balanced_basis.target_width

    @property
    def result_sha256(self) -> str:
        return str(self.receipt["result_sha256"])

    @property
    def source_projector_sha256(self) -> str:
        return str(self.receipt["source_projector_sha256"])

    def verify(self) -> "Lfm2CircuitBalancedCoordinateEvidence":
        """Reconstruct the typed object so every tensor and seal is rechecked."""

        return Lfm2CircuitBalancedCoordinateEvidence(
            balanced_basis=self.balanced_basis,
            source_projector=self.source_projector,
            completion_weight=self.completion_weight,
            receipt=self.receipt,
        )

    def as_dict(self) -> dict[str, Any]:
        return dict(self.receipt)


def build_lfm2_circuit_balanced_coordinate_evidence(
    parent_model: torch.nn.Module,
    trace_cache: Any,
    target_width: int,
    *,
    foundation_interface: TiedCausalInterfaceFold,
    tokenizer_bridge: StrictTokenizerBridgeArtifact,
    circuit_graph: ModelCircuitGraph | None = None,
    validation_fraction: float = 0.25,
    split_seed: str = "lfm2-circuit-balanced-v1",
    max_rows_per_block: int = 512,
    vocabulary_margin_top_k: int = 8,
    vocabulary_margin_floor_fraction: float = 0.05,
    completion_weight: float = 0.35,
) -> Lfm2CircuitBalancedCoordinateEvidence:
    """Build a vocabulary-observable projector from weights and sealed cache only."""

    target_width = _positive_int(target_width, "target_width")
    completion_weight = _unit_interval(completion_weight, "completion_weight")
    if not isinstance(tokenizer_bridge, StrictTokenizerBridgeArtifact):
        raise Lfm2CircuitBalancedCoordinateEvidenceError(
            "tokenizer_bridge must be a StrictTokenizerBridgeArtifact."
        )
    tokenizer_bridge.verify_files()
    interface_report = _sealed_interface_report(foundation_interface)
    if (
        foundation_interface.target_width != target_width
        or foundation_interface.vocabulary_size
        != tokenizer_bridge.target_vocabulary_size
    ):
        raise Lfm2CircuitBalancedCoordinateEvidenceError(
            "The foundation interface does not span the dynamic target width and "
            "complete bridged vocabulary."
        )
    result = build_lfm2_circuit_balanced_basis(
        parent_model,
        trace_cache,
        target_width,
        validation_fraction=validation_fraction,
        split_seed=split_seed,
        max_rows_per_block=max_rows_per_block,
        circuit_graph=circuit_graph,
        vocabulary_margin_observability=True,
        vocabulary_margin_top_k=vocabulary_margin_top_k,
        vocabulary_margin_floor_fraction=vocabulary_margin_floor_fraction,
    )
    if result.source_width != foundation_interface.source_width:
        raise Lfm2CircuitBalancedCoordinateEvidenceError(
            "Balanced parent width differs from the foundation producer width."
        )
    result_report = result.as_dict()
    result_body = dict(result_report)
    result_seal = _digest(
        result_body.pop("result_sha256", None),
        "balanced basis result SHA-256",
    )
    if canonical_sha256(result_body) != result_seal:
        raise Lfm2CircuitBalancedCoordinateEvidenceError(
            "The circuit-balanced basis result seal is invalid."
        )
    structural_roles = frozenset(value.role for value in result.inventory.operators)
    missing_roles = sorted(_REQUIRED_STRUCTURAL_ROLES.difference(structural_roles))
    circuit_ir = result_report.get("balanced_basis", {}).get("circuit_ir", {})
    margin_report = result_report.get("vocabulary_margin_observability")
    if (
        missing_roles
        or not isinstance(circuit_ir, Mapping)
        or circuit_ir.get("residual_width_verified") is not True
        or circuit_ir.get("all_writer_nodes_covered") is not True
        or circuit_ir.get("all_output_interfaces_covered") is not True
        or not isinstance(margin_report, Mapping)
        or margin_report.get("enabled") is not True
        or margin_report.get("full_parent_logits_read_from_sealed_cache") is not True
        or margin_report.get("parent_forward_calls") != 0
        or margin_report.get("trace_regeneration_calls") != 0
    ):
        raise Lfm2CircuitBalancedCoordinateEvidenceError(
            "Circuit-balanced evidence does not cover the complete residual-facing "
            "structure, IR writers, output interface, and vocabulary margins."
        )
    basis = _matrix(result.basis, "balanced basis")
    projector = (basis @ basis.T).contiguous()
    identity = torch.eye(target_width, dtype=torch.float64)
    orthonormal_error = float((basis.T @ basis - identity).abs().max().item())
    projector_symmetry_error = float((projector - projector.T).abs().max().item())
    projector_idempotence_error = float(
        (projector @ projector - projector).abs().max().item()
    )
    if max(
        orthonormal_error,
        projector_symmetry_error,
        projector_idempotence_error,
    ) > 1e-10:
        raise Lfm2CircuitBalancedCoordinateEvidenceError(
            "Circuit-balanced basis is not an RMSNorm-safe orthogonal isometry."
        )
    foundation_table_sha256 = _causal_table_sha256(
        foundation_interface.target_weight
    )
    foundation_table_coordinate_sha256 = _tensor_sha256(
        foundation_interface.target_weight
    )
    foundation_encoder_sha256 = _tensor_sha256(
        foundation_interface.producer_encoder
    )
    body: dict[str, Any] = {
        "schema": LFM2_CIRCUIT_BALANCED_COORDINATE_EVIDENCE_SCHEMA,
        "recipe": LFM2_CIRCUIT_BALANCED_COORDINATE_RECIPE,
        "status": "authenticated-orthogonal-source-projector-ready",
        "source_width": result.source_width,
        "target_width": result.target_width,
        "completion_weight": completion_weight,
        "basis_sha256": result.map_sha256,
        "balanced_basis_result_sha256": result_seal,
        "source_projector_sha256": _tensor_sha256(projector),
        "source_projector_rank": target_width,
        "required_source_projector_rank": target_width,
        "structural_roles": sorted(structural_roles),
        "required_structural_roles": sorted(_REQUIRED_STRUCTURAL_ROLES),
        "all_required_structural_roles_present": True,
        "attention_short_conv_mlp_state_norm_and_output_observable": True,
        "all_ir_writer_nodes_observable": True,
        "all_ir_output_interfaces_observable": True,
        "vocabulary_margin_observability_enabled": True,
        "vocabulary_margin_report_sha256": margin_report["report_sha256"],
        "circuit_graph_sha256": result.circuit_graph.fingerprint,
        "inventory_sha256": result.inventory.manifest["inventory_sha256"],
        "cache_key": result_report["cache"]["cache_key"],
        "cache_corpus_sha256": result_report["cache"]["corpus_sha256"],
        "cache_parent_identity_sha256": result_report["cache"][
            "parent_identity_sha256"
        ],
        "foundation_interface": {
            "report_sha256": interface_report["report_sha256"],
            "producer_encoder_sha256": foundation_encoder_sha256,
            "structural_target_table_sha256": foundation_table_sha256,
            "structural_target_table_coordinate_sha256": (
                foundation_table_coordinate_sha256
            ),
            "source_width": foundation_interface.source_width,
            "target_width": foundation_interface.target_width,
            "target_vocabulary_row_count": foundation_interface.vocabulary_size,
            "producer_encoder_preserved": True,
            "structural_target_table_preserved": True,
            "one_global_encoder_dual_decoder_gauge": True,
        },
        "token_bridge": {
            "artifact_sha256": tokenizer_bridge.artifact_sha256,
            "profile_fingerprint": tokenizer_bridge.profile_fingerprint,
            "donor_tokenizer_semantics_sha256": (
                tokenizer_bridge.donor_tokenizer_semantics_sha256
            ),
            "target_tokenizer_semantics_sha256": (
                tokenizer_bridge.target_tokenizer_semantics_sha256
            ),
            "full_target_vocabulary_row_count": (
                tokenizer_bridge.target_vocabulary_size
            ),
            "full_target_vocabulary_row_coverage": True,
        },
        "rmsnorm_isometry_contract": {
            "orthonormal_basis_isometry": True,
            "basis_transpose_is_left_inverse": True,
            "source_projector_is_symmetric": True,
            "source_projector_is_idempotent": True,
            "orthogonal_projector_only_crosses_norm_boundary": True,
            "nonorthogonal_balanced_factors_cross_rmsnorm": False,
            "nonorthogonal_factors_absorbed_into_adjacent_linear_operators": True,
            "balanced_encoder_decoder_retained_for_audit_only": True,
            "orthonormality_max_abs_error": orthonormal_error,
            "projector_symmetry_max_abs_error": projector_symmetry_error,
            "projector_idempotence_max_abs_error": projector_idempotence_error,
        },
        "causal_output_contract": {
            "full_structural_table_is_foundation": True,
            "partial_causal_trie_may_supply_residual_only": True,
            "partial_causal_trie_may_replace_structural_table": False,
            "unsupported_logit_policy_required": (
                LFM2_UNSUPPORTED_LOGIT_POLICY_ADAPTIVE_STRONGEST_SURVIVING_BACKGROUND_SEAL
            ),
            "fully_unsupported_rows_must_equal_structural_baseline": True,
            "supported_rows_may_retain_target_dtype_nullspace_roundoff": True,
            "supported_row_off_support_drift_is_advisory": True,
        },
        "evidence_policy": {
            "complete_parent_weights_read_without_execution": True,
            "sealed_parent_cache_only": True,
            "full_parent_logits_read_from_sealed_cache": True,
            "heldout_used_for_selection": False,
            "parent_forward_calls": 0,
            "parent_generate_calls": 0,
            "parent_trace_regeneration_calls": 0,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "autograd_used": False,
            "pruning_used": False,
            "quantization_used": False,
            "random_completion_used": False,
        },
    }
    receipt = {**body, "result_sha256": canonical_sha256(body)}
    return Lfm2CircuitBalancedCoordinateEvidence(
        balanced_basis=result,
        source_projector=projector,
        completion_weight=completion_weight,
        receipt=receipt,
    )


__all__ = [
    "LFM2_CIRCUIT_BALANCED_COORDINATE_EVIDENCE_SCHEMA",
    "LFM2_CIRCUIT_BALANCED_COORDINATE_RECIPE",
    "Lfm2CircuitBalancedCoordinateEvidence",
    "Lfm2CircuitBalancedCoordinateEvidenceError",
    "build_lfm2_circuit_balanced_coordinate_evidence",
]
